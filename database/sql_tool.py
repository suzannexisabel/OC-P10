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
Chaque ligne de stats représente un joueur sur toute la saison, pas un match.
Les nombres de points, de tirs, de rebonds et de passes sont des TOTAUX ;
minutes_per_game et les pourcentages sont des MOYENNES/POURCENTAGES.
Les pourcentages sont enregistrés sur 100 (37.5 veut dire 37,5 %).
La colonne three_points_made vient d'un en-tête Excel corrompu « 15:00 » :
son identification comme 3PM est probable mais les 3PM/3PA/3P% de la source
ne concordent pas toujours. Utiliser le three_point_pct fourni pour le classement.
"""

EXAMPLES = """Question: Quels joueurs ont marqué le plus de points sur la saison ?
SQL: SELECT p.player_name, s.points_total FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' ORDER BY s.points_total DESC LIMIT 5

Question: Qui a le meilleur pourcentage de réussite à 3 points sur la saison ?
SQL: SELECT p.player_name, s.three_point_pct, s.three_points_attempted FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.three_points_attempted > 0 ORDER BY s.three_point_pct DESC LIMIT 5

Question: Quels joueurs ont la meilleure moyenne de points par match, avec au moins 20 matchs ?
SQL: SELECT p.player_name, ROUND(1.0 * s.points_total / s.games_played, 2) AS points_per_game, s.games_played FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.games_played >= 20 ORDER BY points_per_game DESC LIMIT 5

Question: Quels sont les meilleurs pourcentages à 3 points avec au moins 100 tentatives ?
SQL: SELECT p.player_name, s.three_point_pct, s.three_points_attempted FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' AND s.three_points_attempted >= 100 ORDER BY s.three_point_pct DESC LIMIT 5

Question: Quel est le total de points des joueurs de chaque équipe dans ce fichier ?
SQL: SELECT p.team_code, SUM(s.points_total) AS total_points_des_joueurs FROM stats AS s JOIN players AS p ON p.player_id = s.player_id JOIN reports AS r ON r.report_id = s.report_id WHERE r.season = '2024-25' GROUP BY p.team_code ORDER BY total_points_des_joueurs DESC LIMIT 10
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