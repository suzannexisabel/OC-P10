# SportSee — Assistant NBA avec Mistral

Ce projet implémente un assistant conversationnel sur la NBA, accessible via une interface Streamlit. Il combine une approche Retrieval-Augmented Generation (RAG) pour exploiter des discussions Reddit et un outil SQL pour interroger les statistiques des joueurs de la saison régulière 2024-25.

La recherche documentaire utilise des embeddings Mistral et un index FAISS pour retrouver les passages utiles à la question. Les données statistiques, importées depuis un fichier Excel dans une base SQLite, permettent de réaliser des classements, des filtres et des calculs.

Un agent Pydantic AI utilisant le modèle Mistral génère les réponses et peut appeler l’outil SQL selon la demande. Pydantic valide les entrées et les sorties du pipeline, tandis que Logfire permet d’observer son exécution.

Le projet comprend également un jeu de questions annotées, des scripts d’évaluation RAGAS et un notebook d’analyse pour comparer les différentes versions de l’assistant.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Modules principaux](#modules-principaux)
- [Limites connues](#limites-connues)

## Fonctionnalités

- 🔍 **Recherche documentaire avec FAISS** : récupération de passages issus des documents Reddit pour répondre aux questions sur les opinions et les débats autour de la NBA.
- 📊 **Interrogation des statistiques avec SQL** : consultation des données de la saison régulière NBA 2024-25 dans SQLite pour effectuer des classements, des filtres et des calculs.
- 🤖 **Génération de réponses avec Mistral et Pydantic AI** : un agent utilise les passages documentaires et peut appeler l’outil SQL selon la question.
- ✅ **Validation des données avec Pydantic** : contrôle des questions, des réponses et des données importées depuis Excel.
- 💬 **Interface conversationnelle Streamlit** : saisie des questions et affichage des échanges.
- 🔎 **Observabilité avec Logfire** : traces des étapes du pipeline et des appels de l’agent pour analyser son exécution.
- 📈 **Évaluation avec RAGAS** : comparaison des versions sur un jeu de questions annotées, selon cinq métriques : Faithfulness, Context Precision, Context Recall, Response Relevancy et Factual Correctness.
- ⚙️ **Configuration personnalisable** : modèle Mistral, taille et chevauchement des passages documentaires, nombre de passages récupérés.


## Prérequis

- **Python 3.11.16** : version utilisée pour le développement et les évaluations du projet.
- **Git** : pour cloner le dépôt.
- **Une clé API Mistral** : disponible depuis [la console Mistral](https://console.mistral.ai/). Elle est utilisée pour la génération des réponses, les embeddings, la génération des requêtes SQL et les évaluations RAGAS.
- **Une connexion Internet** : nécessaire pour les appels à l’API Mistral et l’envoi des traces à Logfire.
- **Un compte et une authentification Logfire** : pour enregistrer et consulter les traces d’exécution.
- **Les données sources dans `inputs/`** : les quatre documents Reddit au format PDF et le fichier `regular NBA.xlsx`, nécessaires pour reconstruire l’index FAISS et la base SQLite.

Les dépendances Python sont listées dans `requirements.txt` et installées lors de l’étape suivante.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/suzannexisabel/OC-P10.git
cd OC-P10
```

Les commandes suivantes sont à exécuter depuis la racine du projet.

### 2. Créer et activer un environnement virtuel

Utiliser Python 3.11.16, version employée pour le projet.

```bash
python3.11 -m venv venv
```

Sur macOS/Linux :

```bash
source venv/bin/activate
```

Sur Windows, dans PowerShell :

```powershell
.\venv\Scripts\Activate.ps1
```

Vérifier la version de Python :

```bash
python --version
```

### 3. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

Vérifier la compatibilité des dépendances installées :

```bash
python -m pip check
```

### 4. Configurer la clé API Mistral

Créer un fichier `.env` à la racine du projet :

```dotenv
MISTRAL_API_KEY=votre_clé_api_mistral
```

Cette clé est utilisée par l’application, l’indexation documentaire, l’outil SQL et les évaluations RAGAS.

### 5. Configurer Logfire

Logfire permet de consulter les traces d’exécution du pipeline et de l’agent Pydantic AI.

Depuis la racine du projet, se connecter à son compte et sélectionner un projet Logfire accessible :

```bash
logfire auth
logfire projects use <nom-du-projet>
```

Le projet utilisé pendant le développement est `oc-p10`. Remplacer `<nom-du-projet>` par le nom de son propre projet pour reproduire cette configuration.

L’application envoie les traces sous le nom de service `oc-p10-rag`. Les identifiants locaux stockés dans `.logfire/` ne doivent pas être versionnés.


## Structure du projet

```text
.
├── MistralChat.py                      # Interface Streamlit et pipeline RAG avec agent SQL
├── indexer.py                          # Construction de l’index documentaire FAISS
├── inputs/                             # Données sources
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
├── data/
│   └── nba.sqlite                      # Base SQLite des statistiques NBA
├── database/
│   ├── schema.sql                      # Définition des tables SQLite
│   ├── load_excel_to_db.py             # Validation et import des données Excel
│   └── sql_tool.py                     # Génération, contrôle et exécution des requêtes SQL
├── utils/
│   ├── config.py                       # Configuration des modèles et du pipeline
│   ├── data_loader.py                  # Chargement des documents sources
│   ├── schemas.py                      # Schémas de validation Pydantic
│   └── vector_store.py                 # Gestion de l’index FAISS et recherche documentaire
├── vector_db/
│   ├── document_chunks.pkl             # Passages documentaires et métadonnées
│   └── faiss_index.idx                 # Index des embeddings
├── evaluation/
│   ├── questions.json                  # Jeu de questions et réponses de référence
│   ├── evaluate_ragas.py               # Génération des réponses et évaluation RAGAS
│   ├── add_context_recall.py           # Ajout de Context Recall aux résultats existants
│   ├── nootbook_analyses_ragas.ipynb   # Analyse comparative des versions
│   ├── results/                        # Résultats des évaluations
│   │   ├── v0.1_baseline_ragas_results.csv
│   │   ├── v0.2_pydantic_ai_results.csv
│   │   └── v0.3_sql_results.csv
│   └── logfire/                        # Captures des traces des huit questions évaluées
├── EVOLUTIONS.md                       # Documentation technique des évolutions 
├── README.md                           # Documentation du fonctionnement et de l’installation
├── requirements.txt                    # Dépendances Python
├── .env                                # Variables d’environnement locales
├── .gitignore                          # Fichiers exclus du versionnement
└── .logfire/                           # Configuration et authentification locales Logfire
 
```

Le fichier `.env`,  et le dossier `.logfire/` sont propres à l’environnement local et ne doivent pas être versionnés.


## Utilisation

Exécuter les commandes suivantes depuis la racine du projet, avec l’environnement virtuel activé et la configuration terminée.

### 1. Préparer les données sources

Le dossier `inputs/` doit contenir :

- Les quatre documents `Reddit 1.pdf` à `Reddit 4.pdf`.
- Le fichier de statistiques `regular NBA.xlsx`.

Les PDF contiennent les discussions utilisées pour les questions d’opinion. Le fichier Excel contient les statistiques des joueurs de la saison régulière NBA 2024-25.

Vous pouvez ajouter  d'autres documents il suffit de les placer dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec `mistral-embed`
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

Les fichiers produits sont :

- `vector_db/faiss_index.idx`
- `vector_db/document_chunks.pkl`

Relancer cette commande pour reconstruire l’index après une modification des documents sources.

### 3. Importer les statistiques dans SQLite

Pour créer et alimenter une nouvelle base :

```bash
python database/load_excel_to_db.py "inputs/regular NBA.xlsx" --season 2024-25
```

Le script utilise `database/schema.sql` pour créer les tables dans `data/nba.sqlite`. Il valide les données Excel avant leur insertion et importe les statistiques de l’onglet `Données NBA`, ainsi que les informations et les liens des joueurs et équipes.

Cette étape est nécessaire si la base n’est pas déjà fournie et alimentée. Le script refuse un nouvel import du même fichier pour la même saison et le même type de saison.

La base contient des statistiques agrégées par joueur pour la saison régulière 2024-25. Elle ne contient pas de statistiques match par match ni de distinction domicile/extérieur.

### 4. Lancer l’application

```bash
streamlit run MistralChat.py
```

L’application est accessible à l’adresse [http://localhost:8501](http://localhost:8501).

Saisir une question dans l’interface :

- Pour les opinions et débats, l’agent utilise les passages documentaires récupérés par FAISS.
- Pour les statistiques, l’agent peut appeler l’outil SQL afin d’interroger SQLite.
- Pour une question mixte, la réponse distingue les résultats statistiques des éléments documentaires.

La recherche FAISS est exécutée avant l’appel à l’agent, même pour une question uniquement statistique. Dans ce cas, les instructions demandent à l’agent de répondre à partir des résultats SQL.

### 5. Exécuter les évaluations RAGAS

Le fichier `evaluation/questions.json` contient les questions du jeu d’évaluation et leurs réponses de référence.

Pour évaluer la version actuelle du pipeline :

```bash
python evaluation/evaluate_ragas.py
```

Pour un essai limité à une question :

```bash
python evaluation/evaluate_ragas.py --limit 1
```

Le script appelle le pipeline de l’application, récupère les réponses et les contextes, puis calcule quatre métriques : Faithfulness, Context Precision, Response Relevancy et Factual Correctness.

Les résultats sont sauvegardés après chaque question dans :

```text
evaluation/results/v0.3_sql_results.csv
```

**Attention : chaque exécution remplace ce fichier, même avec `--limit`. Sauvegarder les résultats existants avant de relancer l’évaluation.**

#### Ajouter Context Recall

Context Recall est calculée séparément à partir des questions, des références et des contextes déjà enregistrés dans les trois fichiers de résultats :

```bash
python evaluation/add_context_recall.py --files \
  evaluation/results/v0.1_baseline_ragas_results.csv \
  evaluation/results/v0.2_pydantic_ai_results.csv \
  evaluation/results/v0.3_sql_results.csv
```

Cette commande est présentée pour macOS/Linux.

Le script ajoute la colonne `context_recall` sans régénérer les réponses ni modifier les autres valeurs. Il crée une sauvegarde `.bak` de chaque fichier modifié et ignore les fichiers qui possèdent déjà cette colonne.

#### Consulter l’analyse

Ouvrir `evaluation/nootbook_analyses_ragas.ipynb` dans un environnement compatible avec les notebooks Jupyter pour consulter l’analyse comparative des versions.

Les évaluations utilisent l’API Mistral. Les réponses et les scores peuvent varier entre deux exécutions ; reproduire la procédure ne garantit donc pas des résultats strictement identiques.


## Modules principaux

### `MistralChat.py`

Contient l’interface Streamlit et le pipeline de réponse :
- Validation de la question avec Pydantic.
- Recherche des passages documentaires dans FAISS.
- Construction du prompt et appel de l’agent Pydantic AI.
- Appel de l’outil SQL par l’agent selon la demande.
- Récupération des résultats SQL dans les contextes utilisés pour l’évaluation.
- Validation de la réponse et affichage des échanges.
- Instrumentation des étapes avec Logfire.

### `indexer.py`

Orchestre la construction de l’index documentaire : chargement des sources, découpage, génération des embeddings et sauvegarde de l’index FAISS.

### `utils/data_loader.py`

Charge les fichiers sources et extrait leur contenu :
- Lecture des PDF, avec recours à l’OCR si l’extraction textuelle est insuffisante ou échoue.
- Lecture des fichiers TXT, DOCX, CSV et Excel.
- Création d’un document par feuille pour les classeurs Excel comportant plusieurs feuilles.
- Validation des documents et de leurs métadonnées avec Pydantic.

### `utils/vector_store.py`

Gère les passages documentaires et l’index FAISS :
- Découpage des documents avec conservation des métadonnées.
- Génération des embeddings Mistral par lots.
- Construction d’un index `IndexFlatIP` avec normalisation des vecteurs pour la similarité cosinus.
- Sauvegarde et chargement de l’index et des passages.
- Recherche des passages les plus proches de la question.
- Validation des passages, des embeddings et des résultats de recherche.

### `utils/schemas.py`

Définit les schémas Pydantic des documents, métadonnées, passages, embeddings, résultats de recherche, questions et réponses.

Ces validations contrôlent la structure des données et certaines valeurs invalides, comme les textes vides ou les embeddings contenant des valeurs non finies. Elles ne garantissent pas l’exactitude factuelle des réponses.

### `utils/config.py`

Centralise la configuration de l’application. Les principaux paramètres personnalisables sont :

| Paramètre | Valeur actuelle | Rôle |
|---|---|---|
| `MODEL_NAME` | `ministral-3b-2512` | Modèle utilisé par l’agent et le générateur SQL de l’application. |
| `EMBEDDING_MODEL` | `mistral-embed` | Modèle utilisé pour vectoriser les documents et les questions. |
| `CHUNK_SIZE` | `1500` | Taille cible des passages documentaires, en caractères. |
| `CHUNK_OVERLAP` | `150` | Chevauchement entre les passages, en caractères. |
| `EMBEDDING_BATCH_SIZE` | `32` | Nombre de passages envoyés par lot à l’API d’embeddings. |
| `SEARCH_K` | `5` | Nombre maximal de passages récupérés pour une question. |
| `INPUT_DIR` | `inputs` | Dossier des documents sources. |
| `VECTOR_DB_DIR` | `vector_db` | Dossier de sauvegarde de l’index et des passages. |
| `APP_TITLE` | `NBA Analyst AI` | Titre affiché dans Streamlit. |
| `NAME` | `NBA` | Nom utilisé dans les textes de l’interface. |

Après une modification du modèle d’embeddings ou des paramètres de découpage, reconstruire l’index avec `python indexer.py`.

La clé `MISTRAL_API_KEY` est chargée depuis le fichier `.env`.

Les modèles utilisés pour l’évaluation RAGAS sont définis séparément dans les scripts d’évaluation. Modifier `MODEL_NAME` ici ne change donc pas le modèle juge.

### `database/schema.sql`

Définit les tables, les relations et les contraintes de la base SQLite :
- `players` : identité, équipe et liens des joueurs.
- `reports` : saison, type de saison et provenance de l’import.
- `stats` : statistiques des joueurs associées à un rapport.
- `matches` : structure prévue pour des matchs individuels, non alimentée avec les données actuelles.

### `database/load_excel_to_db.py`

Valide et importe les données Excel dans SQLite :
- Vérification des onglets, des en-têtes, des valeurs statistiques et des hyperliens.
- Validation des joueurs et des statistiques avec Pydantic.
- Contrôle des doublons et de la cohérence entre matchs joués, victoires et défaites.
- Insertion dans une transaction, annulée en cas d’erreur.

### `database/sql_tool.py`

Transforme une question en requête SQL avec Mistral, puis retourne la requête et les lignes obtenues :
- Transmission du schéma et du dictionnaire des statistiques au modèle.
- Exécution de requêtes de lecture uniquement.
- Ouverture de SQLite en lecture seule.
- Limitation des résultats retournés à 20 lignes.
- Interruption des requêtes SQL dépassant environ trois secondes.
- Détection de certaines demandes nécessitant des données indisponibles, notamment les statistiques match par match ou domicile/extérieur.

### `evaluation/evaluate_ragas.py`

Exécute le pipeline sur le jeu de questions annotées, calcule quatre métriques RAGAS et sauvegarde les réponses, les contextes et les scores dans un fichier CSV.

### `evaluation/add_context_recall.py`

Calcule Context Recall à partir des contextes et des références déjà enregistrés dans les fichiers CSV, puis ajoute cette métrique en conservant les autres valeurs.

### `evaluation/nootbook_analyses_ragas.ipynb`

Présente l’analyse comparative des versions, les tableaux de résultats et les graphiques des métriques.

## Limites connues

- **Périmètre statistique** : les données fournies concernent uniquement la saison régulière NBA 2024-25. Elles sont agrégées par joueur et ne permettent pas d’analyser les derniers matchs ou de distinguer les performances à domicile et à l’extérieur.

- **Couverture documentaire** : FAISS récupère un nombre limité de passages selon leur similarité avec la question. Ces passages peuvent être incomplets ou peu pertinents et ne permettent pas de garantir un comptage exhaustif des mentions dans les documents Reddit.

- **Qualité des sources** : l’extraction des PDF et l’OCR peuvent introduire des erreurs. Certaines données Excel présentent également des ambiguïtés, notamment la colonne interprétée comme le nombre de tirs à trois points réussis.

- **Fiabilité des réponses** : les validations Pydantic contrôlent la structure des données, mais ne garantissent ni la pertinence des passages, ni la justesse des requêtes SQL générées, ni l’exactitude factuelle des réponses.

- **Limites de l’outil SQL** : les résultats retournés sont limités à 20 lignes et les requêtes SQL dépassant environ trois secondes sont interrompues.

- **Portée des évaluations** : le jeu annoté comprend huit questions. Les résultats décrivent les performances sur ces cas et ne suffisent pas à établir la qualité du système pour toutes les questions NBA. Les métriques qui échouent restent manquantes et doivent être distinguées des scores nuls.

- **Variabilité** : les réponses et les scores RAGAS peuvent varier entre deux exécutions. La reproduction de l’environnement et de la procédure ne garantit pas des résultats strictement identiques.
