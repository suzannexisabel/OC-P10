"""Tool LangChain : question NBA -> SQL SQLite -> lignes vérifiées.

Dépendances : langchain-core (installé avec langchain), mistralai, pydantic.
Exemple d'intégration :
    from sql_tool import MistralSQLGenerator, build_sql_tool
    sql_tool = build_sql_tool(MistralSQLGenerator(), "data/nba.sqlite")
    result = sql_tool.invoke({"question": "Quels sont les 5 meilleurs marqueurs en 2024-25 ?"})
"""
from __future__ import annotations

import re
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SQLQuestion(BaseModel):
    question: str = Field(min_length=3, description="Question chiffrée sur les statistiques NBA 2024-25")

class MistralSQLGenerator:
    """Adaptateur minimal pour utiliser le SDK Mistral."""

    def __init__(self, model: str = "ministral-3b-2512"):
        from dotenv import load_dotenv
        from mistralai.client import Mistral

        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY manque dans l'environnement ou le fichier .env")
        self.client = Mistral(api_key=api_key)
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

# Les libellés définissent le sens des colonnes. Ils sont transmis au LLM.
SCHEMA = """SQLite ; tables :
players(player_id PK, player_name UNIQUE, player_url, team_code, team_name, team_url, age)

reports(report_id PK, season, season_type, source_file, imported_at, description)

stats(stat_id PK, player_id FK -> players, report_id FK -> reports,
 games_played, wins, losses, minutes_per_game, points_total,
 field_goals_made, field_goals_attempted, field_goal_pct,
 three_points_made, three_points_attempted, three_point_pct,
 free_throws_made, free_throws_attempted, free_throw_pct,
 offensive_rebounds, defensive_rebounds, total_rebounds, assists,
 turnovers, steals, blocks, personal_fouls, fantasy_points,
 double_doubles, triple_doubles, plus_minus, offensive_rating,
 defensive_rating, net_rating, assist_pct, assist_turnover_ratio,
 assist_ratio, offensive_rebound_pct, defensive_rebound_pct,
 total_rebound_pct, turnover_ratio, effective_field_goal_pct,
 true_shooting_pct, usage_pct, pace, player_impact_estimate, possessions)

matches(match_id PK, match_date, season, home_team, away_team, home_score, away_score) : VIDE.

Il n'y a qu'un rapport : saison 2024-25, saison régulière.
Chaque ligne de stats représente les statistiques d'un joueur
sur la saison, pas les statistiques d'un match individuel.
La table matches est vide.

Unités :
- Les points, tirs, rebonds, passes décisives, balles perdues,
  interceptions, contres et fautes sont des totaux de saison.
- minutes_per_game est une moyenne par match.
- Les pourcentages sont enregistrés sur 100 :
  37.5 signifie 37,5 %. Ne pas multiplier ces valeurs par 100.
- Les ratios et les statistiques par 100 possessions
  ne sont pas des totaux de saison.

Dictionnaire des colonnes :

Identité et liens :
- player_id : identifiant interne du joueur dans la base.
- player_name : nom du joueur.
- player_url : URL de la page du joueur enregistrée dans la base.
  À récupérer pour une demande de lien vers ce joueur
  ou vers ses statistiques.
- team_code : code de l'équipe.
- team_name : nom de l'équipe.
- team_url : URL de la page de l'équipe enregistrée dans la base.
- age : âge du joueur enregistré dans la source.

Rapport :
- report_id : identifiant du rapport de statistiques.
- season : saison concernée.
- season_type : type de saison.
- source_file : nom du fichier source.
- imported_at : date d'importation dans la base,
  pas la date d'un match ni une garantie d'actualisation des données.
- description : description du rapport.

Participation :
- stat_id : identifiant de la ligne de statistiques.
- games_played : nombre de matchs joués.
- wins : nombre de victoires lors des matchs joués.
- losses : nombre de défaites lors des matchs joués.
- minutes_per_game : minutes moyennes jouées par match.

Points et tirs :
- points_total : total de points marqués sur la saison.
- field_goals_made : total de tirs de champ réussis,
  comprenant les tirs à 2 et à 3 points.
- field_goals_attempted : total de tirs de champ tentés.
- field_goal_pct : pourcentage de réussite aux tirs de champ.
- three_points_made : total présumé de tirs à 3 points réussis.
  Cette colonne provient d'un en-tête source corrompu.
  Son identification est probable et certaines valeurs
  ne concordent pas avec les tentatives et le pourcentage.
  Ne pas l'utiliser pour recalculer three_point_pct.
- three_points_attempted : total de tirs à 3 points tentés.
- three_point_pct : pourcentage de réussite à 3 points.
  Utiliser directement cette colonne pour un classement
  de réussite à 3 points.
- free_throws_made : total de lancers francs réussis.
- free_throws_attempted : total de lancers francs tentés.
- free_throw_pct : pourcentage de réussite aux lancers francs.

Statistiques de jeu :
- offensive_rebounds : total de rebonds offensifs.
- defensive_rebounds : total de rebonds défensifs.
- total_rebounds : total de rebonds offensifs et défensifs réunis.
- assists : total de passes décisives.
- turnovers : total de balles perdues.
- steals : total d'interceptions.
- blocks : total de contres.
- personal_fouls : total de fautes personnelles.
- fantasy_points : points fantasy selon le barème de la source.
  Ne pas les confondre avec les points réellement marqués.
- double_doubles : nombre de matchs avec au moins 10
  dans deux catégories parmi les points, rebonds,
  passes décisives, interceptions et contres.
- triple_doubles : nombre de matchs avec au moins 10
  dans trois de ces catégories.
- plus_minus : écart de score en faveur ou en défaveur
  de l'équipe lorsque le joueur est sur le terrain.
  Ne pas le confondre avec net_rating.

Statistiques avancées :
- offensive_rating : points marqués par l'équipe
  par 100 possessions lorsque le joueur est sur le terrain.
- defensive_rating : points encaissés par l'équipe
  par 100 possessions lorsque le joueur est sur le terrain.
  Une valeur plus basse signifie moins de points encaissés.
- net_rating : différence entre offensive_rating
  et defensive_rating.
- assist_pct : pourcentage des paniers de ses coéquipiers
  auxquels le joueur contribue par une passe décisive
  lorsqu'il est sur le terrain.
- assist_turnover_ratio : ratio passes décisives
  sur balles perdues. Ce n'est pas un pourcentage.
- assist_ratio : passes décisives par 100 possessions.
- offensive_rebound_pct : pourcentage de rebonds offensifs
  disponibles captés lorsque le joueur est sur le terrain.
- defensive_rebound_pct : pourcentage de rebonds défensifs
  disponibles captés lorsque le joueur est sur le terrain.
- total_rebound_pct : pourcentage de rebonds disponibles
  captés lorsque le joueur est sur le terrain.
  Ce n'est pas un nombre de rebonds ni une moyenne par match.
- turnover_ratio : balles perdues par 100 possessions.
- effective_field_goal_pct : pourcentage de réussite aux tirs
  ajusté pour tenir compte de la valeur supérieure des tirs à 3 points.
- true_shooting_pct : mesure de l'efficacité au scoring
  tenant compte des tirs à 2 points, à 3 points et des lancers francs.
- usage_pct : pourcentage d'utilisation offensive du joueur.
  Ne pas le confondre avec son temps de jeu.
- pace : rythme de jeu exprimé en possessions par 48 minutes.
- player_impact_estimate : indicateur global de l'impact du joueur.
- possessions : nombre total de possessions jouées.

Correspondances avec les questions :
- « le plus de matchs » -> games_played.
- « le plus de points sur la saison » -> points_total.
- « moyenne de points par match » ->
  1.0 * points_total / NULLIF(games_played, 0).
- « le plus de rebonds » -> total_rebounds.
- « meilleur pourcentage de rebonds » -> total_rebound_pct.
- « pourcentage de rebonds offensifs » -> offensive_rebound_pct.
- « pourcentage de rebonds défensifs » -> defensive_rebound_pct.
- « meilleur pourcentage à 3 points » -> three_point_pct.
- « meilleur ratio passes / pertes de balle » -> assist_turnover_ratio.
- « lien vers les statistiques du joueur » -> player_url.
- « lien vers l'équipe » -> team_url.

Règles d'interprétation :
- Respecter la statistique et le nombre de joueurs demandés.
- Ne pas ajouter de statistiques ou de classements non demandés.
- Ne jamais remplacer un pourcentage par une quantité.
- Utiliser directement les pourcentages enregistrés.
- Pour calculer une moyenne par match à partir d'un total,
  diviser par NULLIF(games_played, 0).
- Ne pas ajouter de seuil de matchs ou de tentatives
  si l'utilisateur n'en demande pas.
- Pour une demande de lien, sélectionner le nom et l'URL
  enregistrée correspondante. Ne jamais fabriquer une URL.
- Une URL enregistrée est une information accessible via SQL ;
  sa récupération ne nécessite pas de recherche sur Internet.
"""

EXAMPLES = """Question: Quels sont les 5 joueurs ayant la meilleure moyenne de points par match, avec au moins 20 matchs joués ?
SQL: SELECT p.player_name, s.points_total AS points_per_game, s.games_played FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.games_played >= 20 AND s.points_total IS NOT NULL ORDER BY s.points_total DESC, p.player_name ASC LIMIT 5

Question: Quels sont les 3 joueurs ayant joué le plus de matchs ?
SQL: SELECT p.player_name, s.games_played FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.games_played IS NOT NULL ORDER BY s.games_played DESC, p.player_name ASC LIMIT 3

Question: Quels sont les 5 joueurs ayant le meilleur pourcentage à 3 points avec au moins 3 tentatives par match ?
SQL: SELECT p.player_name, s.three_point_pct, s.three_points_attempted AS three_point_attempts_per_game FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.three_points_attempted >= 3 AND s.three_point_pct IS NOT NULL ORDER BY s.three_point_pct DESC, p.player_name ASC LIMIT 5

Question: Quels sont les 4 joueurs ayant le meilleur pourcentage de rebonds ?
SQL: SELECT p.player_name, s.total_rebound_pct FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.total_rebound_pct IS NOT NULL ORDER BY s.total_rebound_pct DESC, p.player_name ASC LIMIT 4

Question: Quels sont les 5 joueurs ayant le meilleur ratio de passes décisives sur balles perdues ?
SQL: SELECT p.player_name, s.assist_turnover_ratio FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.assist_turnover_ratio IS NOT NULL ORDER BY s.assist_turnover_ratio DESC, p.player_name ASC LIMIT 5

Question: Quels sont les 5 joueurs ayant le meilleur defensive rating avec au moins 20 matchs joués ?
SQL: SELECT p.player_name, s.defensive_rating, s.games_played FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.games_played >= 20 AND s.defensive_rating IS NOT NULL ORDER BY s.defensive_rating ASC, p.player_name ASC LIMIT 5

Question: Compare les moyennes de points, les minutes par match et le pourcentage de réussite à 3 points de Stephen Curry et de Nikola Jokić.
SQL: SELECT p.player_name, s.points_total AS points_per_game, s.minutes_per_game, s.three_point_pct FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND p.player_name IN ('Stephen Curry', 'Nikola Jokić') ORDER BY p.player_name ASC LIMIT 2

Question: Donne-moi le lien vers les statistiques du joueur ayant la meilleure moyenne de points par match.
SQL: SELECT p.player_name, s.points_total AS points_per_game, p.player_url FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.points_total IS NOT NULL ORDER BY s.points_total DESC, p.player_name ASC LIMIT 1
"""

PROMPT = """Tu génères UNE requête SQLite SELECT pour répondre à la question.
Retourne uniquement le SQL brut, sans Markdown ni explication.
N'utilise que les colonnes ci-dessous, les jointures nécessaires et LIMIT 20 au plus.
Ne crée jamais de données de match individuel. Ne déduis pas une période récente
à partir d'une saison agrégée. Ne rajoute pas de seuil de tentatives si la
question ne l'indique pas ; expose alors three_points_attempted dans les résultats.
N'utilise pas three_points_made pour recalculer three_point_pct.
Ne prétends pas que la somme des points des joueurs d'une équipe représente
les points réellement marqués par cette équipe (transferts et périodes inconnus).

Identifie précisément la statistique demandée à partir du dictionnaire.
Sélectionne uniquement les informations nécessaires à la question.

Respecte le nombre de joueurs demandé : top 4 signifie LIMIT 4.
Si aucun nombre n'est précisé pour un classement, utilise LIMIT 5.

Pour un classement, trie d'abord par la statistique demandée,
puis par p.player_name ASC pour départager les égalités de manière stable.
Exclus les valeurs NULL de la statistique classée.

Ne rajoute pas de statistiques ou de classements non demandés.
Pour un pourcentage stocké, utilise directement la colonne correspondante.

{schema}
Exemples :
{examples}
Question: {question}
SQL:"""

# Questions impossibles avec les seules statistiques agrégées du classeur.
UNAVAILABLE = re.compile(
    r"\b(?:derniers?\s+\d+\s+matchs?|\d+\s+derniers?\s+matchs?|"
    r"matchs?\s+individuels?|"
    r"domicile|ext[eé]rieur|home|away|last\s+\d+\s+games?|"
    r"derni[eè]res?\s+rencontres?)\b", re.IGNORECASE,
)


def _clean_sql(content: Any) -> str:
    text = content.content if hasattr(content, "content") else content
    if not isinstance(text, str):
        raise ValueError("Le modèle n'a pas renvoyé une requête SQL textuelle.")
    text = text.strip()
    match = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip().rstrip(";").strip()


def execute_select(db_path: str | Path, sql: str) -> dict[str, Any]:
    """Exécute une seule requête, dans SQLite en lecture seule et sans PRAGMA/ATTACH."""
    path = Path(db_path).expanduser().resolve(strict=True)
    sql = _clean_sql(sql)
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.I):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")
    uri = path.as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True, timeout=2) as conn:
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(
            lambda action, _a, _b, _db, _trigger: sqlite3.SQLITE_OK
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)
            else sqlite3.SQLITE_DENY
        )
        deadline = time.monotonic() + 3
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cursor = conn.execute(sql)  # sqlite3 refuse plusieurs instructions.
        if cursor.description is None:
            raise ValueError("La requête doit retourner des lignes.")
        columns = [c[0] for c in cursor.description]
        rows = [dict(row) for row in cursor.fetchmany(21)]
        return {"sql": sql, "columns": columns, "rows": rows[:20],
                "truncated": len(rows) > 20, "count": min(len(rows), 20)}


def answer_sql_question(question: str, llm: Any, db_path: str | Path = "data/nba.sqlite") -> dict[str, Any]:
    """Génère, exécute et retourne la requête ; aucune réponse chiffrée inventée."""
    if not question or len(question.strip()) < 3:
        raise ValueError("La question doit contenir au moins trois caractères.")
    if UNAVAILABLE.search(question):
        return {"status": "unavailable", "reason": "La base contient seulement des statistiques agrégées sur la saison 2024-25 ; les statistiques par match et domicile/extérieur ne sont pas disponibles.", "sql": None, "rows": []}
    prompt = PROMPT.format(schema=SCHEMA, examples=EXAMPLES, question=question)
    sql = _clean_sql(llm.invoke(prompt))
    try:
        result = execute_select(db_path, sql)
    except sqlite3.Error as exc:
        raise ValueError(f"La requête SQL générée est invalide ou non autorisée : {exc}") from exc
    return {"status": "ok", **result}


def build_sql_tool(llm: Any, db_path: str | Path = "data/nba.sqlite"):
    """Construit un Tool LangChain réutilisable par l'agent"""
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=lambda question: answer_sql_question(question, llm, db_path),
        name="nba_season_sql",
        description=("Interroge les statistiques chiffrées des joueurs NBA sur la saison "
                     "régulière 2024-25. Retourne le SQL et les lignes de la base. "
                     "Signale l'absence de données par match ou domicile/extérieur."),
        args_schema=SQLQuestion,
    )