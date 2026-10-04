"""Ajoute Context Recall aux CSV existants sans modifier leurs données."""

import argparse
import ast
import asyncio
import csv
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import MISTRAL_API_KEY

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def create_ragas_llm():
    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY est absente.")

    judge = ChatOpenAI(
        model="ministral-3b-2512",
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
        temperature=0,
        max_retries=3,
    )
    return LangchainLLMWrapper(judge)


def parse_contexts(value):
    try:
        contexts = json.loads(value)
    except json.JSONDecodeError:
        contexts = ast.literal_eval(value)

    if not isinstance(contexts, list) or not all(
        isinstance(context, str) for context in contexts
    ):
        raise ValueError("Les contextes doivent être une liste de textes.")

    return contexts


def read_csv_bytes(raw):
    # utf-8-sig accepte aussi les CSV avec une marque BOM.
    text = raw.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text, newline="")))


def save_csv(path, original_bytes, original_rows, scores):
    """Vérifie les anciennes cellules avant de remplacer le fichier."""
    updated_rows = [
        original_rows[0] + ["context_recall"],
        *[
            row + [score]
            for row, score in zip(original_rows[1:], scores)
        ],
    ]

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerows(updated_rows)

    encoding = (
        "utf-8-sig"
        if original_bytes.startswith(b"\xef\xbb\xbf")
        else "utf-8"
    )
    updated_bytes = buffer.getvalue().encode(encoding)

    # Vérification : toutes les anciennes cellules sont identiques.
    checked_rows = read_csv_bytes(updated_bytes)
    if [row[:-1] for row in checked_rows] != original_rows:
        raise RuntimeError("Vérification échouée : aucune modification faite.")

    # Évite d'écraser une modification faite pendant l'évaluation.
    if path.read_bytes() != original_bytes:
        raise RuntimeError(f"{path.name} a changé pendant le calcul.")

    backup = path.with_name(path.name + ".bak")
    # N'écrase jamais une sauvegarde existante.
    with backup.open("xb") as file:
        file.write(original_bytes)

    # Écrit d'abord un fichier temporaire, puis remplace le CSV.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(updated_bytes)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"CSV mis à jour : {path.name}")
    print(f"Sauvegarde : {backup.name}")


async def main(paths):
    # Valide tous les fichiers avant de commencer les appels au modèle.
    prepared = []

    for path in paths:
        raw = path.read_bytes()
        rows = read_csv_bytes(raw)

        if not rows:
            raise ValueError(f"{path.name} est vide.")

        header = rows[0]
        required = {"question", "reference", "retrieved_contexts"}

        if not required.issubset(header):
            raise ValueError(f"Colonnes nécessaires absentes : {path.name}")

        if len(header) != len(set(header)):
            raise ValueError(f"Colonnes en double : {path.name}")

        if any(len(row) != len(header) for row in rows[1:]):
            raise ValueError(f"Ligne CSV mal formée : {path.name}")

        if "context_recall" in header:
            print(f"{path.name} : colonne déjà présente, fichier ignoré.")
            continue

        backup = path.with_name(path.name + ".bak")
        if backup.exists():
            raise FileExistsError(
                f"{backup.name} existe déjà. "
                "Déplace cette sauvegarde avant de relancer."
            )

        prepared.append((path, raw, rows))

    if not prepared:
        return

    metric = LLMContextRecall(llm=create_ragas_llm())

    for path, raw, rows in prepared:
        print(f"\n--- {path.name} ---", flush=True)
        header = rows[0]
        scores = []

        for number, row in enumerate(rows[1:], start=1):
            record = dict(zip(header, row))
            print(
                f"Question {number}/{len(rows) - 1} : "
                f"{record['question']}",
                flush=True,
            )

            try:
                if not record["question"].strip():
                    raise ValueError("Question vide.")
                if not record["reference"].strip():
                    raise ValueError("Référence vide.")

                sample = SingleTurnSample(
                    user_input=record["question"],
                    reference=record["reference"],
                    retrieved_contexts=parse_contexts(
                        record["retrieved_contexts"]
                    ),
                )

                score = float(
                    await metric.single_turn_ascore(sample)
                )

                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Score invalide : {score}")

                scores.append(str(score))
                print(f"Context Recall : {score:.4f}", flush=True)

            except Exception as error:
                # Conserve la ligne et laisse seulement le nouveau score vide.
                scores.append("")
                print(
                    f"Échec : {error}. Context Recall laissé vide.",
                    flush=True,
                )

        save_csv(path, raw, rows, scores)

        valid_scores = [float(score) for score in scores if score != ""]
        print(f"Scores calculés : {len(valid_scores)}/{len(scores)}")
        if valid_scores:
            print(f"Moyenne : {sum(valid_scores) / len(valid_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="Chemins des CSV à enrichir.",
    )
    args = parser.parse_args()

    paths = args.files or sorted(
        RESULTS_DIR.glob("v0.*_results.csv")
    )
    paths = [path.resolve() for path in paths]

    if len(paths) != 3 or len(set(paths)) != 3:
        parser.error(
            "Il faut exactement trois CSV distincts. "
            "Utilise --files pour préciser leurs chemins."
        )

    asyncio.run(main(paths))