# MistralChat.py (version RAG)
import streamlit as st
import os
import logging
import logfire

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from dotenv import load_dotenv

from pydantic import ValidationError
from utils.schemas import RAGRequest, RAGResponse

from database.sql_tool import (MistralSQLGenerator, 
                               build_sql_tool, 
                               UNAVAILABLE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

# --- Configuration de l'observabilité Logfire ---
logfire.configure(
    service_name="oc-p10-rag",
)

logfire.instrument_pydantic_ai()

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()


# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    mistral_model = MistralModel(
        model,
        provider=MistralProvider(api_key=api_key),
    )

    agent = Agent(
        mistral_model,
        output_type=str,
        model_settings={"temperature": 0.1},
        instructions="""
        Tu es NBA Analyst AI. Réponds précisément à la demande.

        Pour les statistiques chiffrées de la saison 2024-25 et les liens
        enregistrés dans la base, appelle nba_season_sql avant de répondre.
        Transmets une question fidèle à la demande, sans ajouter de métriques.

        Si le tool retourne status="ok", utilise les lignes retournées.
        Ne les qualifie pas de simulations, d'estimations ou de données
        incomplètes sans élément explicite qui le démontre.
        Respecte les noms des colonnes et leurs unités.
        N'ajoute aucun classement non demandé.
        Si plusieurs joueurs ont la même valeur, tu peux signaler leur égalité.

        Pour les opinions et débats, utilise uniquement les passages
        documentaires qui concernent réellement le joueur ou le sujet demandé.
        Un passage récupéré ne constitue pas automatiquement une preuve pertinente.
        N'attribue pas à un joueur des propos concernant un autre joueur.
        Si les passages ne permettent pas de répondre, dis-le brièvement.

        Pour une question d'opinion, ne commente pas l'absence de statistiques
        et ne propose pas spontanément une recherche chiffrée.

        Pour une question mixte, distingue les statistiques SQL des opinions.

        Mentionne une limite uniquement lorsqu'elle empêche de répondre
        à une partie de la demande. Ne termine pas chaque réponse
        par une remarque générale sur les limites des données.

        Si l'utilisateur demande un lien, récupère player_url ou team_url
        avec nba_season_sql et présente l'URL retournée sous forme de lien
        Markdown. Si aucune URL n'est renseignée, indique-le brièvement.

        N'affiche des liens que si l'utilisateur en demande explicitement.
        Pour un classement sans demande de lien, présente uniquement
        le tableau avec les joueurs et les statistiques demandées.
        """
    )

    logging.info("Agent Pydantic AI initialisé avec Mistral.")

except Exception as e:
    st.error(
        "Erreur lors de l'initialisation de "
        f"l'agent Pydantic AI : {e}"
    )
    logging.exception(
        "Erreur lors de l'initialisation "
        "de l'agent Pydantic AI"
    )
    st.stop()

def get_sql_tool():
    generor = MistralSQLGenerator(model=MODEL_NAME)
    return build_sql_tool(generor, "data/nba.sqlite")

@agent.tool_plain
def nba_season_sql(question: str) -> dict:
    """Interroge les statistiques NBA de la saison 2024-25.
    
    À utiliser pour les claseements totaux, moyennes et pourcentages.
    La base ne continet pas de statistiques par match.
    """
    with logfire.span("Exécution du Tool SQL NBA", question=question):
        return get_sql_tool().invoke({"question": question})

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Prompt Système pour RAG --- 
SYSTEM_PROMPT =  """Tu es NBA Analyst AI, un assistant sur la NBA.

Si la question porte uniquement sur des statistiques chiffrées,
appelle nba_season_sql et réponds uniquement à partir des lignes retournées.
Ignore les contextes documentaires pour cette réponse.
Ne commente pas les onglets Excel, les valeurs manquantes ou la provenance
des lignes, sauf si la question le demande explicitement.

Pour les questions sur les opinions et débats, utilise les contextes
documentaires ci-dessous. Pour une question mixte, distingue les résultats
SQL des commentaires.

Si le Tool indique que les données sont indisponibles, explique la limite
sans remplacer les données demandées par des statistiques de saison.

Rédige toujours une réponse en phrases complètes.
Pour un classement de plusieurs joueurs, présente les résultats dans un
tableau avec les colonnes « Rang », « Joueur » et « Valeur ».
Indique la statistique mesurée, son unité et la saison.
Pour un seul joueur, réponds en une phrase complète avec la valeur et son unité.
Ne recopie pas simplement une liste de valeurs brutes.

CONTEXTES DOCUMENTAIRES :
---
{context_str}
---

QUESTION DU FAN :
{question}

RÉPONSE :"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---

def generer_reponse(prompt: str) -> str:
    """
    Envoie le prompt enrichi à l'agent Pydantic AI.
    """
    if not prompt or not prompt.strip():
        logging.warning(
            "Tentative de génération avec un prompt vide."
        )
        return "Je ne peux pas traiter une demande vide."

    try:
        logging.info(
            "Appel de l'agent Pydantic AI avec "
            f"le modèle '{model}'."
        )

        result = agent.run_sync(prompt)

        logging.info(
            "Réponse reçue de l'agent Pydantic AI."
        )

        return result.output

    except Exception as e:
        st.error(
            "Erreur lors de l'appel à "
            f"l'agent Pydantic AI : {e}"
        )
        logging.exception(
            "Erreur pendant l'exécution "
            "de l'agent Pydantic AI"
        )

        return (
            "Je suis désolé, une erreur technique "
            "m'empêche de répondre. "
            "Veuillez réessayer plus tard."
        )

@logfire.instrument(
    "Exécution du pipeline RAG",
)
def executer_rag(
    question: str,
) -> tuple[str, list[str]]:
    """
    Exécute le pipeline RAG utilisé par Streamlit
    et retourne la réponse et les contextes.
    """

    #Valider la question reçue
    with logfire.span(
        "Validation Pydantic de la question"
    ):
        validated_request = RAGRequest(
            question=question
        )

    question = validated_request.question

    if UNAVAILABLE.search(question):
        return (
            "Cette demande nécessite des statistiques match par match "
            "ou une distinction domicile/extérieur. Ma base contient "
            "uniquement des statistiques agrégées sur la saison 2024-25.",
            [],
        )

    # Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error(
            "Le service de recherche de connaissances "
            "n'est pas disponible. "
            "Impossible de traiter votre demande."
        )
        logging.error(
            "VectorStoreManager non disponible "
            "pour la recherche."
        )
        st.stop()

    # Rechercher le contexte dans le Vector Store
    try:
        logging.info(
            f"Recherche de contexte pour la question: "
            f"'{question}' avec k={SEARCH_K}"
        )

        with logfire.span(
            "Recherche FAISS"
        ):
            search_results = vector_store_manager.search(
                question,
                k=SEARCH_K,
            )

        logging.info(
            f"{len(search_results)} chunks trouvés "
            "dans le Vector Store."
        )

    except Exception as e:
        st.error(
            "Une erreur est survenue lors de la "
            f"recherche d'informations pertinentes: {e}"
        )

        logging.exception(
            "Erreur pendant vector_store_manager.search "
            f"pour la query: {question}"
        )

        search_results = []

    # Conserver les contextes pour RAGAS
    retrieved_contexts = [
        result["text"]
        for result in search_results
    ]

    # Formater le contexte pour le prompt LLM
    context_str = "\n\n---\n\n".join(
        [
            (
                f"Source: "
                f"{result['metadata'].get('source', 'Inconnue')} "
                f"(Score: {result['score']:.1f}%)\n"
                f"Contenu: {result['text']}"
            )
            for result in search_results
        ]
    )

    if not search_results:
        context_str = (
            "Aucune information pertinente trouvée "
            "dans la base de connaissances "
            "pour cette question."
        )

        logging.warning(
            f"Aucun contexte trouvé pour la query: "
            f"{question}"
        )

    # Construire exactement le même prompt
    with logfire.span(
        "Construction du promt RAG",
        nombre_chunks=len(search_results)
    ):
        final_prompt_for_llm = SYSTEM_PROMPT.format(
            context_str=context_str,
            question=question,
        )

    # Utiliser la fonction Mistral existante
    response_content = generer_reponse(
        final_prompt_for_llm
    )

    # Valider la reponse et les contextes
    with logfire.span(
        "Validation Pydantic de la reponse"
    ):
        validated_response = RAGResponse(
            answer=response_content,
            retrieved_contexts=retrieved_contexts,
        )

    return (validated_response.answer,
            validated_response.retrieved_contexts,
    )

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        response_content, _ = executer_rag(prompt)

        # Affichage de la réponse complète
        message_placeholder.write(response_content)

    # 3. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")