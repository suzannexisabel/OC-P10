# Évolutions du projet SportSee

Ce document présente les évolutions du projet dans l’ordre chronologique, avec les problèmes rencontrés, les modifications réalisées, leurs justifications et les limites constatées.

## 1. Adaptation du modèle et mise en place de l’évaluation de référence — V01

**Références Git :**
- `b6cedea` : changement du modèle Mistral.
- `4610ab5` : extraction du pipeline pour rendre ses sorties accessibles à l’évaluation.
- `5757281` — tag `v0.1-baseline` : ajout de l’évaluation RAGAS de référence.

### Problème et objectif

Le modèle initial `mistral-small-latest` ne fonctionnait pas avec le compte utilisé et bloquait l’utilisation de l’application. La cause précise du blocage n’a pas été confirmée.

Par ailleurs, la logique RAG était directement intégrée à l’interface Streamlit. Il était nécessaire de rendre la réponse et les contextes récupérés accessibles à un script d’évaluation afin d’établir une référence pour les comparaisons ultérieures.

### Changements réalisés

**Adaptation du modèle**

Dans `utils/config.py`, `mistral-small-latest` a été remplacé par `ministral-3b-2512`. Le modèle d’embeddings `mistral-embed` est resté inchangé.

**Refactorisation du pipeline**

La fonction `executer_rag(question)` a été créée dans `MistralChat.py`. Elle regroupe la recherche FAISS, la préparation des contextes, la construction du prompt et l’appel à la génération.

Elle retourne deux éléments :
- La réponse générée.
- La liste des passages documentaires récupérés.

L’interface Streamlit appelle désormais cette fonction. La recherche et la construction du prompt existantes ont été reprises dans cette fonction.

**Évaluation RAGAS**

Le script `evaluation/evaluate_ragas.py` appelle la même fonction que l’application et évalue ses sorties avec quatre métriques :

- Faithfulness.
- Context Precision, avec `LLMContextPrecisionWithoutReference`.
- Response Relevancy.
- Factual Correctness.

Le modèle juge est `ministral-3b-2512`, configuré avec une température de `0`. Response Relevancy utilise également les embeddings `mistral-embed`. Ces appels passent par l’API Mistral.

Le jeu `evaluation/questions.json` contient huit cas annotés : deux questions textuelles, quatre questions statistiques, une question bruitée et une question hors périmètre. Chaque cas comporte un identifiant, une catégorie, une question et une réponse de référence. Les cas sont validés avec Pydantic avant leur utilisation.

Les réponses, les contextes et les scores sont sauvegardés après chaque question dans `evaluation/results/v0.1_baseline_ragas_results.csv`. En cas d’échec du calcul d’une métrique, son score reste manquant et l’évaluation continue.

Le commit ajoute également un notebook d’analyse et les dépendances suivantes à `requirements.txt` : NumPy, Pydantic, RAGAS, LangChain Community, LangChain OpenAI et le SDK OpenAI.

### Justification

Le changement de modèle répond à un blocage d’utilisation. Il ne constitue pas une amélioration de qualité démontrée par une comparaison entre modèles.

La refactorisation permet à l’interface et à l’évaluation d’utiliser une fonction commune, sans recopier le pipeline dans le script RAGAS.

Les quatre métriques apportent des mesures complémentaires pour examiner la fidélité aux contextes, leur précision, la pertinence des réponses et leur conformité aux références.

### Validation et conclusion de l’évaluation V01

L’évaluation de référence met en évidence une fidélité limitée aux contextes (Faithfulness moyenne : **0,26**) et une faible précision des passages récupérés (Context Precision moyenne : **0,07**). Les réponses enregistrées comportent des développements non justifiés par les sources et des écarts aux réponses attendues.

Ces observations motivent la poursuite du travail sur l’exploitation des contextes, la fiabilité des réponses statistiques et le respect du périmètre des questions. La refactorisation rend le pipeline évaluable, mais ne constitue pas à elle seule une amélioration de sa qualité.

L’analyse détaillée des réponses et des métriques est disponible dans [le notebook d’analyse](evaluation/nootbook_analyses_ragas.ipynb). Context Recall a été ajoutée ultérieurement aux résultats V01, sans régénérer les réponses.

### Limites

- Le jeu de huit questions constitue une première référence, avec une couverture limitée des usages.
- Les réponses de référence influencent l’évaluation ; leur exactitude ne peut pas être établie à partir du seul fichier de questions.
- Le modèle juge appartient à la même famille et porte le même nom que le modèle de génération. Ses scores ne remplacent pas une vérification humaine.
- Les métriques non calculées doivent être distinguées des scores nuls. Les moyennes excluent les valeurs manquantes.
- Context Recall n’était pas calculée à cette étape. Elle a été ajoutée ultérieurement aux résultats enregistrés.
- Chaque nouvelle exécution du script remplace le CSV de sortie.

## 2. Ajout des validations Pydantic dans le pipeline

**Références Git :**
- `f8c3b01` : validation des documents.
- `9cde344` : validation des passages documentaires.
- `72dbc00` : validation des résultats de recherche.
- `f50a5de` : validation des entrées et sorties du RAG.
- `d9b6944` : validation des embeddings.
- `ee6ba05` : fusion des validations Pydantic dans la branche principale.

### Problème et objectif

L’intégration de Pydantic répond à une exigence du projet et à la volonté de contrôler les structures de données échangées entre les étapes du pipeline.

La V01 utilisait déjà Pydantic pour valider les cas d’évaluation. Cette évolution étend les validations aux documents, aux passages, aux embeddings, aux résultats de recherche et aux entrées et sorties du RAG.

### Changements réalisés

**Création des schémas**

Le fichier `utils/schemas.py` a été ajouté pour définir les modèles suivants :

| Modèle | Contrôles ajoutés |
|---|---|
| `DocumentMetadata` | Champs de provenance obligatoires et non vides ; nom de feuille facultatif. |
| `SourceDocument` | Contenu textuel non vide et métadonnées conformes au schéma. |
| `DocumentChunk` | Identifiant et texte non vides, métadonnées sous forme de dictionnaire. |
| `SearchResult` | Texte non vide, scores finis et bornés selon les valeurs de similarité attendues. |
| `EmbeddedChunk` | Identifiant non vide, vecteur non vide et absence de valeurs NaN ou infinies. |
| `RAGRequest` | Suppression des espaces en début et fin de question ; refus d’une question vide. |
| `RAGResponse` | Réponse non vide après suppression des espaces périphériques et contextes sous forme de liste de textes. |

**Validation pendant le chargement et la recherche**

Dans `utils/data_loader.py`, les documents extraits sont validés avant leur ajout à la liste des sources. Les documents invalides sont ignorés et une erreur est enregistrée dans les logs.

Dans `utils/vector_store.py`, les passages et les résultats de recherche sont également validés. Les éléments invalides sont ignorés avec un message d’erreur.

La génération des embeddings vérifie désormais que le nombre de vecteurs reçus correspond au nombre de passages du lot. Un échec de cette vérification ou de la validation Pydantic interrompt la génération en retournant `None`.

**Validation des échanges du RAG**

Dans `MistralChat.py`, la question est validée avant la recherche. La réponse et les contextes sont validés avant leur retour par `executer_rag`.

Les objets validés sont reconvertis en dictionnaires lorsque les traitements existants attendent ce format.

### Justification

La centralisation des schémas rend explicites les structures attendues et les contraintes appliquées aux données.

Ces contrôles permettent de détecter certaines anomalies au moment où les données entrent dans une étape du pipeline, plutôt que de les laisser se propager jusqu’à la génération ou à l’évaluation.

Cette évolution vise le contrôle des données et la robustesse du code. Elle ne constitue pas une méthode de correction des erreurs factuelles du modèle.

### Validation et conclusion

Les différentes fonctions ont été exécutées manuellement avec des données valides et invalides afin de vérifier le comportement des validations.

L’indexation complète du corpus n’a pas été relancée à cette étape. Ces vérifications ne constituent donc pas une validation de bout en bout de la reconstruction de l’index.

Une évaluation RAGAS supplémentaire a ensuite été réalisée après l’intégration de Pydantic AI. Elle sera présentée dans l’étape suivante : elle ne permet pas d’isoler l’influence des seules validations Pydantic.

### Limites

- Les schémas contrôlent la structure et certaines valeurs, sans garantir la pertinence des passages ni l’exactitude des réponses.
- Les métadonnées des passages et des résultats de recherche restent des dictionnaires dont le contenu n’est pas entièrement contraint.
- La validation des embeddings ne vérifie pas leur qualité sémantique et n’impose pas une dimension fixe.
- Les validations ajoutées à la construction ne revalident pas automatiquement les passages déjà sauvegardés lors du chargement de l’index.
- Les essais manuels ne sont pas conservés sous forme de tests automatisés dans le dépôt.

## 3. Intégration de Pydantic AI — V02

**Références Git :**
- `972d252` : mise à jour du SDK Mistral et adaptation des appels d’embeddings.
- `14812d6` : intégration de Pydantic AI dans le pipeline RAG.
- `526f7a1` : ajout de l’évaluation RAGAS V02.
- `fa6f748` — tag `v0.2-pydantic-ai` : fusion de l’intégration dans la branche principale.

### Problème et objectif

L’intégration de Pydantic AI répond à une exigence du projet. L’objectif est de remplacer l’appel direct au modèle Mistral par un agent, tout en conservant la recherche documentaire et les validations Pydantic précédemment ajoutées.

### Changements réalisés

**Mise à jour des dépendances**

Dans `requirements.txt` :
- `mistralai==0.4.2` est remplacé par `mistralai==2.10.1`.
- `pydantic-ai-slim[mistral]==2.45.0` est ajouté.

Dans `utils/vector_store.py`, le client `MistralClient` est remplacé par `Mistral`. Les appels d’embeddings utilisent désormais `embeddings.create()` avec l’argument `inputs`.

La gestion des erreurs est adaptée à `errors.MistralError`. Lorsqu’une erreur API survient pendant la génération des embeddings, la fonction retourne désormais `None`, au lieu de poursuivre avec des vecteurs nuls pour ce lot. Le traitement générique des autres exceptions conserve toutefois un mécanisme de remplacement par des vecteurs nuls.

**Remplacement de l’appel direct par un agent**

Dans `MistralChat.py`, le modèle est configuré avec `MistralModel` et `MistralProvider`, puis transmis à un `Agent`.

L’agent utilise :
- Le modèle défini par `MODEL_NAME`.
- Une température de génération de `0.1`, inchangée par rapport à l’appel précédent.
- Une sortie textuelle avec `output_type=str`.

La fonction `generer_reponse()` reçoit désormais directement le prompt enrichi sous forme de texte. Elle appelle `agent.run_sync(prompt)` et récupère la réponse avec `result.output`.

La recherche FAISS, la construction du prompt et la validation finale avec `RAGResponse` sont conservées. Aucun outil SQL n’est encore intégré à cette étape.

**Évaluation complémentaire**

Le chemin de sortie du script RAGAS est modifié pour enregistrer les résultats dans `evaluation/results/v0.2_pydantic_ai_results.csv`. Dans le script d’évaluation, seul le chemin du fichier de résultats est modifié. Les métriques et la configuration du modèle juge restent inchangées.

### Justification

Pydantic AI fournit un cadre d’exécution pour l’agent chargé de générer les réponses. Cette intégration constitue également la base sur laquelle l’outil SQL sera ajouté ultérieurement.

La mise à jour du SDK Mistral s’accompagne de l’adaptation des appels existants à sa nouvelle interface.

Le modèle et la température étant conservés, cette évolution porte principalement sur l’intégration logicielle. Elle ne constitue pas un changement de modèle destiné à améliorer les performances.

### Validation et conclusion

L’application Streamlit a été relancée aux étapes cruciales de modification afin de vérifier qu’elle produisait toujours des réponses.

Une évaluation RAGAS complémentaire a été réalisée sur les huit questions. Les moyennes de Faithfulness, Context Precision, Response Relevancy et Factual Correctness augmentent par rapport à la V01. Ces variations ne permettent toutefois pas d’attribuer les gains directement à Pydantic AI : la V02 inclut également les validations Pydantic et la mise à jour du SDK.

L’apport principal de cette étape est l’intégration fonctionnelle de l’agent. L’analyse détaillée des résultats reste disponible dans [le notebook d’analyse](evaluation/nootbook_analyses_ragas.ipynb).

### Limites

- La sortie de l’agent reste un texte libre ; elle n’est pas un objet métier structuré généré par Pydantic AI.
- L’intégration de l’agent ne garantit pas l’exactitude factuelle des réponses ni la pertinence des contextes.
- Les vérifications dans Streamlit confirment le fonctionnement sur les essais réalisés.
- Faithfulness et Context Precision disposent chacune de sept scores sur huit en V02.
- Context Recall a été ajoutée ultérieurement aux résultats enregistrés. Sa moyenne diminue par rapport à la V01, ce qui rend l’évolution des métriques contrastée.

## 4. Ajout de l’observabilité Logfire

**Références Git :**
- `2a517a9` : ajout de l’observabilité Logfire au pipeline RAG.
- `5e26ea2` : fusion de l’observabilité dans la branche principale.

### Problème et objectif

L’utilisation de Logfire répond à une attente du projet et au besoin d’observer l’exécution du pipeline pour les analyses ultérieures.

L’objectif est de rendre visibles les principales étapes du traitement et les appels de l’agent Pydantic AI afin de faciliter l’examen du comportement du système.

### Changements réalisés

**Configuration et instrumentation de l’agent**

Dans `MistralChat.py`, Logfire est configuré avec le nom de service `oc-p10-rag`.

L’appel à `logfire.instrument_pydantic_ai()` active l’instrumentation de l’agent.

**Ajout de traces dans le pipeline**

La fonction `executer_rag()` est instrumentée avec une trace globale intitulée `Exécution du pipeline RAG`.

Des traces spécifiques sont ajoutées autour des étapes suivantes :

- Validation Pydantic de la question.
- Recherche FAISS.
- Construction du prompt RAG.
- Validation Pydantic de la réponse.

La trace de construction du prompt reçoit également l’attribut `nombre_chunks`, correspondant au nombre de passages récupérés.

Ces ajouts entourent les traitements existants sans modifier les règles de recherche ou de génération.

**Ajout des dépendances**

Les dépendances suivantes sont ajoutées à `requirements.txt` :

- `logfire==5.1.0`.
- `protobuf==5.29.6`.
- `googleapis-common-protos==1.70.0`.

Les deux dernières sont regroupées dans le fichier sous le commentaire de compatibilité Logfire / Streamlit.

**Configuration locale**

L’authentification a été réalisée avec `logfire auth`, puis le projet `oc-p10` a été sélectionné avec `logfire projects use oc-p10`.

### Justification

Les traces permettent d’examiner les étapes du pipeline et les appels de l’agent au sein d’une même exécution.

Cette visibilité complète l’évaluation RAGAS : les métriques mesurent certains aspects des réponses, tandis que les traces aident à comprendre le déroulement du traitement.

L’observabilité sert ainsi au diagnostic et à l’analyse du système.

### Validation et conclusion

Un test de connexion a été exécuté pour envoyer un événement Logfire depuis l’environnement local.

L’application Streamlit a ensuite été utilisée et la présence des traces a été vérifiée dans Logfire. Ces traces ont également servi aux analyses ultérieures du projet.

Cette étape apporte une visibilité sur l’exécution. Elle ne constitue pas, à elle seule, une amélioration de la qualité des réponses.

### Limites

- La consultation des traces nécessite une authentification et un accès au projet Logfire concerné.
- Les traces ne garantissent ni la pertinence des passages récupérés ni l’exactitude factuelle des réponses.
- Ce commit instrumente le pipeline RAG et l’agent ; l’outil SQL n’est pas encore intégré à cette étape.
- La configuration et les identifiants locaux doivent être recréés dans un nouvel environnement.

## 5. Création de la base SQLite et import des statistiques NBA

**Référence Git :**
- `2ebe113` : ajout du schéma SQLite, du script d’import et de la base NBA.

### Problème et objectif

Cette étape prépare une base structurée pour interroger les statistiques NBA avec SQL, en complément de la recherche documentaire FAISS.

L’objectif est de rendre les données Excel accessibles sous forme de tables pour les futurs filtres, calculs et classements.

### Changements réalisés

**Création du schéma relationnel**

Le fichier `database/schema.sql` définit quatre tables :

- `players` : identité, équipe, âge et liens des joueurs.
- `reports` : saison, type de saison et provenance de l’import.
- `stats` : statistiques associées à un joueur et à un rapport.
- `matches` : structure pour des matchs individuels, non alimentée à cette étape.

Le schéma ajoute des clés étrangères, des contraintes d’unicité et des index. Chaque joueur ne peut avoir qu’une ligne de statistiques par rapport.

**Ajout du script d’import**

Le fichier `database/load_excel_to_db.py` lit les onglets `Données NBA` et `Equipe` du classeur.

Il associe les colonnes Excel aux champs statistiques, récupère les hyperliens des joueurs et équipes et valide les données avec Pydantic.

Les contrôles portent notamment sur :

- Les huit premiers en-têtes de l’onglet `Données NBA`.
- Les informations des joueurs et les hyperliens HTTP(S).
- Les types numériques et les valeurs entières attendues.
- La cohérence entre matchs joués, victoires et défaites.
- L’absence de noms de joueurs en double.
- La présence des colonnes attendues dans la table `stats`.
- L’absence d’un import antérieur du même fichier pour la même saison et le même type de saison.

Les données sont validées avant toute écriture. Les insertions du rapport, des joueurs et des statistiques sont réalisées dans une transaction annulée en cas d’erreur.

**Ajout de la base**

La base `data/nba.sqlite` est ajoutée au dépôt.

### Justification

SQLite a été choisi pour sa simplicité de mise en place et son stockage dans un fichier local, sans serveur de base de données à administrer.

La séparation entre joueurs, rapports et statistiques explicite les relations et la provenance des données. L’import reproductible et les validations permettent de contrôler les données avant leur exploitation par l’outil SQL.

### Validation et conclusion

Le script intègre des contrôles automatiques lors de l’import. Les essais manuels réalisés à l’époque ne sont pas précisément documentés.

Une vérification en lecture seule de la base fournie, réalisée lors de la rédaction de cette documentation, confirme :

- 569 joueurs.
- 569 lignes de statistiques.
- Un rapport pour la saison régulière 2024-25.
- Aucun match individuel enregistré.
- Un contrôle d’intégrité SQLite réussi.
- Aucune violation de clé étrangère, aucun nom de joueur en double et aucune incohérence entre victoires, défaites et matchs joués.

La base est ainsi disponible pour l’intégration de l’outil SQL à l’étape suivante. Ces contrôles ne constituent pas une comparaison exhaustive avec le fichier Excel.

### Limites

- Les données disponibles concernent uniquement la saison régulière 2024-25 et sont agrégées par joueur.
- La table `matches` est vide : les analyses match par match ou domicile/extérieur ne sont pas possibles.
- Le script dépend de la structure du classeur et de la position des colonnes. Seuls les huit premiers en-têtes sont explicitement vérifiés.
- Les contrôles de structure et de cohérence ne garantissent pas l’exactitude de chaque valeur source.
- L’import de plusieurs saisons dans une même base nécessite une adaptation du script pour réutiliser les joueurs déjà enregistrés.


## 6. Création de l’outil SQL et intégration à l’agent

**Références Git :**
- `f529a6d` : création de l’outil SQL pour les statistiques NBA.
- `e296c9b` : intégration à l’agent et adaptation des prompts.

### Problème et objectif

L’objectif est de répondre aux questions statistiques à partir de requêtes SQL sur la base NBA, plutôt qu’à partir des seuls passages récupérés par FAISS.

Les classements, filtres et calculs nécessitent de pouvoir interroger l’ensemble des lignes concernées. Une recherche de passages similaires ne garantit pas cette couverture.

### Changements réalisés

**Création de l’outil SQL**

Le fichier `database/sql_tool.py` ajoute :

- `SQLQuestion` : schéma Pydantic de la question transmise à l’outil.
- `MistralSQLGenerator` : génération d’une requête SQL avec Mistral, à température `0`.
- `answer_sql_question()` : préparation du prompt, génération et exécution de la requête.
- `execute_select()` : exécution contrôlée dans SQLite.
- `build_sql_tool()` : création d’un outil LangChain nommé `nba_season_sql`.

Le prompt transmet au modèle le schéma de la base, les unités, les limites des données et des exemples de requêtes.

L’outil retourne notamment la requête exécutée, les colonnes, les lignes obtenues, leur nombre et une indication précisant si les résultats ont été limités.

**Contrôles d’exécution**

La base est ouverte en lecture seule. Les requêtes doivent commencer par `SELECT` ou `WITH`, et un contrôle SQLite autorise uniquement les opérations de lecture et les fonctions.

L’exécution accepte une seule instruction et doit retourner des lignes. Les résultats transmis sont limités à 20 lignes. Un contrôle de progression interrompt les requêtes dépassant environ trois secondes.

Certaines demandes nécessitant des données match par match ou domicile/extérieur sont détectées par une expression régulière et reçoivent un statut `unavailable`.

**Intégration à Pydantic AI**

Dans `MistralChat.py`, la fonction `nba_season_sql()` est enregistrée auprès de l’agent avec `@agent.tool_plain`. Elle appelle l’outil sur `data/nba.sqlite`.

Une trace Logfire intitulée `Exécution du Tool SQL NBA` est ajoutée autour de son exécution, avec la question comme attribut.

Les instructions demandent à l’agent :

- D’appeler SQL pour les statistiques et les liens enregistrés.
- D’utiliser les passages documentaires pour les opinions et débats.
- De distinguer les deux sources pour une question mixte.
- De respecter les unités et de ne pas ajouter de classements non demandés.
- De récupérer les URL enregistrées uniquement lorsque des liens sont demandés.
- De signaler une insuffisance des données plutôt que d’inventer une réponse.

Une détection des demandes indisponibles est également ajoutée au début du pipeline, avant la recherche FAISS.

**Adaptation des prompts**

Le prompt de génération SQL est enrichi avec un dictionnaire des colonnes, des correspondances entre questions et statistiques, ainsi que des règles de classement et d’interprétation.

Le prompt de réponse demande une présentation en phrases complètes ou en tableau, avec la statistique, son unité et la saison.

### Justification

SQL permet de filtrer, trier et calculer directement sur les données structurées, sans dépendre de la sélection de quelques passages documentaires.

La séparation des consignes vise à utiliser chaque source selon la demande : SQLite pour les statistiques et FAISS pour les discussions.

Les contrôles d’exécution protègent la base contre les écritures et limitent les ressources utilisées par une requête. Le dictionnaire des colonnes aide le modèle à interpréter les données.

### Validation et conclusion

L’outil SQL a été testé séparément, puis utilisé dans l’application Streamlit après son intégration.

Ces essais vérifient son fonctionnement sur les demandes testées. La liste précise des cas et leurs résultats n’est pas conservée dans les éléments fournis.

Cette étape rend les statistiques accessibles à l’agent par SQL. L’évaluation V03 et l’ajout des résultats SQL aux contextes RAGAS seront présentés à l’étape suivante.

### Limites

- Une requête autorisée et exécutable peut rester incorrecte par rapport à la question. Les contrôles techniques ne valident pas son sens.
- Certains exemples ajoutés au prompt confondent des totaux de saison avec des moyennes par match, malgré les indications correctes du dictionnaire. Ces contradictions peuvent orienter le modèle vers une requête erronée.
- L’appel à SQL repose sur les instructions et la décision de l’agent ; il n’est pas imposé par un routage déterministe pour toutes les questions statistiques.
- La recherche FAISS reste exécutée avant l’agent pour les questions statistiques ordinaires. Le pipeline dépend donc encore de la disponibilité de l’index.
- La détection des demandes indisponibles repose sur des formulations prédéfinies et n’est pas exhaustive.
- La limite d’environ trois secondes concerne l’exécution SQLite, pas la durée totale des appels aux modèles.
- Les données restent limitées à la saison régulière 2024-25, avec au maximum 20 lignes retournées par l’outil.

## 7. Évaluation de la version SQL et ajout de Context Recall

**Références Git :**

- `cf30317` — Ajout des résultats SQL aux contextes RAGAS et modification du fichier de sortie de l’évaluation.
- `068534f` — Ajout des résultats et analyses de l’évaluation V03. Tag : `v0.3-sql-rag`.
- `13e9045` — Ajout de la métrique Context Recall et mise à jour de l’analyse. Tag : `v0.3-sql-rag-optimise`.

### Problème et objectif

Après l’intégration de l’outil SQL, les contextes retournés par le pipeline ne contenaient que les passages issus de FAISS. Les résultats SQL utilisés par l’agent pour répondre n’étaient donc pas transmis à RAGAS.

L’objectif était de rendre ces informations accessibles à l’évaluation, puis de compléter la comparaison des versions avec **Context Recall**, qui mesure la couverture des informations de la réponse de référence par les contextes récupérés.

### Changements réalisés

**Prise en compte des résultats SQL**

Dans `MistralChat.py`, la fonction `generer_reponse()` reçoit désormais la liste `retrieved_contexts`.

Après l’exécution de l’agent, elle parcourt les nouveaux messages et récupère les retours de l’outil `nba_season_sql` identifiés par `ToolReturnPart`. Leur contenu est sérialisé en JSON et ajouté aux contextes avec une mention de la source SQL et de la saison 2024-25.

Les résultats proviennent du même appel à l’agent que la réponse évaluée : aucune seconde requête SQL n’est lancée pour reconstruire les contextes.

**Enregistrement de l’évaluation V03**

Dans `evaluation/evaluate_ragas.py`, le fichier de sortie devient `v0.3_sql_results.csv`.

Le commit `068534f` ajoute les résultats V03, met à jour le notebook d’analyse et ajoute huit captures Logfire dans `evaluation/logfire/`.

**Ajout ultérieur de Context Recall**

Le script `evaluation/add_context_recall.py` calcule la métrique `LLMContextRecall` à partir des questions, références et contextes déjà enregistrés dans les trois CSV.

Il utilise `ministral-3b-2512` comme modèle juge, à température `0`. **Il ne relance pas le pipeline RAG et ne régénère pas les réponses.**

Le script ajoute uniquement la colonne `context_recall`. Il vérifie que les anciennes cellules restent identiques, crée une sauvegarde `.bak` et remplace le CSV après écriture dans un fichier temporaire. Les fichiers possédant déjà cette colonne sont ignorés. En cas d’échec sur une question, le nouveau score reste vide.

Le commit met également à jour le notebook d’analyse et ajoute `matplotlib==3.11.2` aux dépendances.

### Justification

Inclure les résultats SQL permet à RAGAS d’évaluer la réponse à partir des informations effectivement reçues par l’agent, y compris celles provenant de la base structurée.

Context Recall complète les métriques initiales en évaluant la couverture des informations attendues. Son calcul sur les résultats sauvegardés conserve les réponses et les contextes des évaluations précédentes.

Le tag `v0.3-sql-rag-optimise` correspond ainsi à un enrichissement de l’évaluation et de l’analyse, sans modification du fonctionnement du système.

### Validation et conclusion

Les traces ont été vérifiées dans Logfire lors des essais. Les résultats V03 contiennent huit cas évalués et la colonne Context Recall ajoutée ultérieurement.

La moyenne de Faithfulness passe d’environ **0,26 en V01 à 0,85 en V03**, avec seulement six scores disponibles sur huit en V03. Context Recall passe de **0,31 à 0,81**.

Ces résultats montrent une meilleure couverture des informations attendues et une meilleure fidélité aux contextes sur les cas mesurés. Toutefois, Context Precision reste limitée à **0,19**, et Factual Correctness reste proche de **0,35** : les progrès ne concernent donc pas toutes les dimensions de qualité.

L’analyse détaillée est présentée dans [le notebook d’analyse](evaluation/nootbook_analyses_ragas.ipynb).

### Limites

- Le jeu d’évaluation contient seulement huit questions.
- Les moyennes doivent être interprétées en tenant compte des scores manquants.
- Les métriques reposent sur un modèle juge et ne remplacent pas une vérification des réponses et des calculs SQL.
- Les contextes V03 mélangent passages FAISS et retours SQL : les scores ne mesurent pas séparément la qualité de chaque source.
- L’ajout de Context Recall enrichit la mesure, mais n’améliore pas à lui seul les réponses du système.

## 8. Mise à jour de la documentation

### Problème et objectif

Le README initial décrivait le prototype et ne correspondait plus au fonctionnement actuel du projet. Il fallait actualiser les instructions de reproduction et documenter les changements réalisés avec leurs justifications.

### Changements réalisés

**Mise à jour de `README.md`**

Le README conserve la structure générale du document initial et décrit désormais :

- Le fonctionnement de l’assistant combinant recherche FAISS et outil SQL.
- Les prérequis, l’installation et la configuration de Mistral et Logfire.
- La reconstruction de l’index documentaire et l’import des statistiques dans SQLite.
- Le lancement de l’application et des évaluations.
- Le rôle des modules et les paramètres personnalisables.
- Les limites connues du système.

Un sommaire a également été ajouté.

**Création de `EVALUATION.md`**

Ce document retrace les évolutions du projet dans l’ordre chronologique. Chaque étape présente les références Git, le problème rencontré, les changements, leur justification, la validation et les limites.

Les conclusions des évaluations y sont résumées. L’analyse détaillée reste disponible dans le notebook dédié.

### Justification

Le README permet de comprendre et de reproduire le fonctionnement du projet. `EVOLUTIONS.md` permet de suivre les décisions techniques et de retrouver les commits associés aux changements.

### Validation et conclusion

La documentation a été rédigée à partir des scripts, des différences Git, des résultats d’évaluation et des essais manuels confirmés.

### Limites

- La documentation décrit la version actuelle et devra être actualisée lors des prochaines modifications.
- Sa rédaction ne constitue pas une validation complète de l’installation dans un nouvel environnement.


## Conclusion générale

Le projet a évolué d’un prototype reposant sur la recherche vectorielle vers un assistant combinant recherche documentaire et interrogation SQL des statistiques NBA.

La validation Pydantic contrôle les données aux différentes étapes du pipeline. Pydantic AI encadre les appels au modèle et l’utilisation de l’outil SQL, tandis que Logfire permet d’observer les exécutions et de les analyser.

Les évaluations montrent une progression de la fidélité aux contextes et de la couverture des informations attendues. Elles mettent également en évidence des limites persistantes, notamment sur la précision des contextes et l’exactitude factuelle des réponses.

Les prochaines améliorations pourraient porter sur la vérification des calculs SQL, la qualité de la recherche documentaire et l’élargissement du jeu d’évaluation. Le README décrit le fonctionnement et les étapes de reproduction du projet.