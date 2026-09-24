PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL UNIQUE,
    player_url TEXT NOT NULL,
    team_code TEXT NOT NULL,
    team_name TEXT NOT NULL,
    team_url TEXT NOT NULL,
    age INTEGER CHECK (age >= 18)
);

CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL,
    source_file TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT,
    season TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,

    CHECK (home_team <> away_team)
);

CREATE TABLE IF NOT EXISTS stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,

    player_id INTEGER NOT NULL,
    report_id INTEGER NOT NULL,

    games_played INTEGER,
    wins INTEGER,
    losses INTEGER,
    minutes_per_game REAL,

    points_total REAL,

    field_goals_made INTEGER,
    field_goals_attempted REAL,
    field_goal_pct REAL,

    three_points_made REAL,
    three_points_attempted REAL,
    three_point_pct REAL,

    free_throws_made REAL,
    free_throws_attempted REAL,
    free_throw_pct REAL,

    offensive_rebounds REAL,
    defensive_rebounds REAL,
    total_rebounds REAL,

    assists REAL,
    turnovers REAL,
    steals REAL,
    blocks REAL,
    personal_fouls REAL,

    fantasy_points REAL,
    double_doubles INTEGER,
    triple_doubles INTEGER,
    plus_minus REAL,

    offensive_rating REAL,
    defensive_rating REAL,
    net_rating REAL,

    assist_pct REAL,
    assist_turnover_ratio REAL,
    assist_ratio REAL,

    offensive_rebound_pct REAL,
    defensive_rebound_pct REAL,
    total_rebound_pct REAL,

    turnover_ratio REAL,
    effective_field_goal_pct REAL,
    true_shooting_pct REAL,
    usage_pct REAL,

    pace REAL,
    player_impact_estimate REAL,
    possessions REAL,

    FOREIGN KEY (player_id)
        REFERENCES players(player_id)
        ON DELETE CASCADE,

    FOREIGN KEY (report_id)
        REFERENCES reports(report_id)
        ON DELETE CASCADE,

    UNIQUE (player_id, report_id)
);

CREATE INDEX IF NOT EXISTS idx_players_team
    ON players(team_code);

CREATE INDEX IF NOT EXISTS idx_stats_player
    ON stats(player_id);

CREATE INDEX IF NOT EXISTS idx_stats_report
    ON stats(report_id);

CREATE INDEX IF NOT EXISTS idx_matches_date
    ON matches(match_date);