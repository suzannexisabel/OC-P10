"""Évaluation RAGAS du pipeline réellement exécuté par MistralChat.py."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
    FactualCorrectness,
)


# Rend la racine du projet importable lorsque le script est lancé directement.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# On importe le pipeline RAG de l'application.
from MistralChat import executer_rag 
from utils.config import MISTRAL_API_KEY  


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "v0.2_pydantic_ai_results.csv"


class EvaluationCase(BaseModel):
    """Structure validée d'un cas du jeu de test annoté."""

    id: str
    category: str
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


def load_test_cases(
    limit: Optional[int] = None,
) -> list[EvaluationCase]:
    """Charge et valide les questions du fichier JSON."""

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    cases = [
        EvaluationCase.model_validate(case)
        for case in raw_cases
    ]

    if limit is not None:
        return cases[:limit]

    return cases


def create_ragas_llm() -> LangchainLLMWrapper:
    """Crée le modèle juge utilisé par les métriques RAGAS."""

    if not MISTRAL_API_KEY:
        raise EnvironmentError(
            "La variable MISTRAL_API_KEY est absente."
        )

    judge = ChatOpenAI(
        model="ministral-3b-2512",
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
        temperature=0,
        max_retries=3,
    )

    return LangchainLLMWrapper(judge)


def create_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Crée le modèle d'embedding utilisé par ResponseRelevancy."""

    if not MISTRAL_API_KEY:
        raise EnvironmentError(
            "La variable MISTRAL_API_KEY est absente."
        )

    embeddings = OpenAIEmbeddings(
        model="mistral-embed",
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
        check_embedding_ctx_length=False,
    )

    return LangchainEmbeddingsWrapper(embeddings)


async def score_metric(
    metric,
    sample: SingleTurnSample,
    metric_name: str,
) -> Optional[float]:
    """Calcule une métrique sans interrompre les autres évaluations."""

    try:
        score = await metric.single_turn_ascore(sample)
        return float(score)

    except Exception:
        logger.exception(
            "Impossible de calculer la métrique %s.",
            metric_name,
        )
        return None


async def evaluate_case(
    case: EvaluationCase,
    faithfulness_metric,
    context_precision_metric,
    response_relevancy_metric,
    factual_correctness_metric,
) -> dict:
    """Appelle le pipeline source puis transmet ses sorties à RAGAS."""

    try:
        # executer_rag() est synchrone : on l'exécute hors de la boucle asyncio.
        answer, retrieved_contexts = await asyncio.to_thread(
            executer_rag,
            case.question,
        )

    except Exception:
        logger.exception(
            "Impossible d'exécuter le RAG pour la question : %s",
            case.question,
        )
        return {
            "case_id": case.id,
            "category": case.category,
            "question": case.question,
            "reference": case.reference_answer,
            "answer": None,
            "retrieved_contexts": None,
            "faithfulness": None,
            "context_precision": None,
            "response_relevancy": None,
            "factual_correctness": None,
        }

    sample = SingleTurnSample(
        user_input=case.question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=case.reference_answer,
    )

    faithfulness = await score_metric(
        faithfulness_metric,
        sample,
        "faithfulness",
    )

    context_precision = await score_metric(
        context_precision_metric,
        sample,
        "context_precision",
    )

    response_relevancy = await score_metric(
    response_relevancy_metric,
    sample,
    "response_relevancy",   
    )

    factual_correctness = await score_metric(
        factual_correctness_metric,
        sample,
        "factual_correctness",
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "question": case.question,
        "reference": case.reference_answer,
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "response_relevancy": response_relevancy,
        "factual_correctness": factual_correctness,
    }


async def main(limit: Optional[int] = None) -> None:
    """Évalue les cas et sauvegarde les résultats progressivement."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases(limit=limit)

    if not test_cases:
        raise ValueError(
            "Aucune question trouvée dans evaluation/questions.json."
        )

    ragas_llm = create_ragas_llm()
    ragas_embeddings = create_ragas_embeddings()

    faithfulness_metric = Faithfulness(
        llm=ragas_llm,
    )

    context_precision_metric = (
        LLMContextPrecisionWithoutReference(
            llm=ragas_llm,
        )
    )

    response_relevancy_metric = ResponseRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    factual_correctness_metric = FactualCorrectness(
        llm=ragas_llm,
    )

    results = []

    for number, case in enumerate(test_cases, start=1):
        print(
            f"\nQuestion {number}/{len(test_cases)} : "
            f"{case.question}"
        )

        result = await evaluate_case(
            case,
            faithfulness_metric,
            context_precision_metric,
            response_relevancy_metric,
            factual_correctness_metric,
        )

        results.append(result)

        # Sauvegarde après chaque question pour ne pas perdre les résultats.
        pd.DataFrame(results).to_csv(
            RESULTS_PATH,
            index=False,
        )

        print("Réponse :", result["answer"])
        print(
            "Faithfulness :", 
            result["faithfulness"]
            )
        print(
            "Context precision :",
            result["context_precision"],
        )
        print(
            "Response relevancy :",
            result["response_relevancy"],
            )
        print(
            "Factual correctness :",
            result["factual_correctness"],
            )

    dataframe = pd.DataFrame(results)

    print("\nRésultats enregistrés dans :", RESULTS_PATH)
    print("\nMoyennes :")
    print(
        dataframe[
            [
                "faithfulness",
                "context_precision",
                "response_relevancy",
                "factual_correctness",
            ]
        ].mean(numeric_only=True)
    )


def parse_args() -> argparse.Namespace:
    """Permet de limiter le nombre de questions pendant les essais."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre de questions à évaluer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(main(limit=arguments.limit))