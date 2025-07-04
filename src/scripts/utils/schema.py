"""
Column normalization and schema enforcement.
"""

import re
from typing import Dict

import pandas as pd

# Keys must be lowercase to match columns after initial cleaning
COLUMN_ALIASES: Dict[str, str] = {
    "playerone": "player_1",
    "playertwo": "player_2",
    "winnername": "winner",
    "prob": "predicted_prob",
    "ev": "expected_value",
    "actual_winner": "winner",
}

SCHEMAS: Dict[str, list] = {
    "matches": [
        "market_id",
        "selection_id",
        "runner_name",
        "ltp",
        "volume",
        "timestamp",
        "match_id",
    ],
    "features": [
        "match_id",
        "player_1",
        "player_2",
        "winner",
        "odds_1",
        "odds_2",
        "implied_prob_1",
        "implied_prob_2",
        "implied_prob_diff",
        "odds_margin",
        "p1_rolling_win_pct",
        "p2_rolling_win_pct",
        "p1_surface_win_pct",
        "p2_surface_win_pct",
    ],
    "value_bets": [
        "match_id",
        "player_1",
        "player_2",
        "odds",
        "predicted_prob",
        "expected_value",
        "kelly_fraction",
        "confidence_score",
        "winner",
    ],
    # --- UPDATED SCHEMAS ---
    "matches_with_ids": [
        "match_id",
        "selection_id",
        "runner_name",
        "ltp",
        "player_1",
        "player_2",
        "winner",
        "selection_id_1",
        "selection_id_2",
        "tourney_date",
        "surface",  # Added
    ],
    "merged_matches": [
        "match_id",
        "selection_id",
        "runner_name",
        "player_1",
        "player_2",
        "winner",
        "selection_id_1",
        "selection_id_2",
        "final_ltp",
        "tourney_date",
        "surface",  # Added
    ],
    # --- END UPDATED SCHEMAS ---
    "predictions": ["match_id", "player_1", "player_2", "predicted_prob", "winner", "odds"],
    "simulations": [
        "match_id",
        "bankroll",
        "kelly_fraction",
        "winner",
        "odds",
        "strategy",
    ],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame columns to pipeline conventions by cleaning names,
    applying aliases, and reordering.
    """
    df = df.copy()

    cleaned_cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = cleaned_cols

    runner_map = {
        col: f"player_{m.group(1)}"
        for col in df.columns
        if (m := re.match(r"runner_(\d+)$", col))
    }
    if runner_map:
        df = df.rename(columns=runner_map)

    existing_aliases = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    if existing_aliases:
        df = df.rename(columns=existing_aliases)

    player_cols = sorted(
        [c for c in df.columns if c.startswith("player_")],
        key=lambda x: int(x.split("_")[1]),
    )
    other_cols = [c for c in df.columns if c not in player_cols]
    df = df[player_cols + other_cols]

    return df


def enforce_schema(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Ensure DataFrame has all columns for the schema.
    Missing columns are added with NaN.
    """
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}")

    df = normalize_columns(df)

    schema_cols = SCHEMAS[schema_name]
    
    for col in schema_cols:
        if col not in df.columns:
            df[col] = pd.NA

    final_cols = [col for col in schema_cols if col in df.columns]
    return df[final_cols]


def patch_winner_column(df: pd.DataFrame, winner_col: str = "winner") -> pd.DataFrame:
    """
    Patch winner column to be 0/1 integer type and fill missing with 0.
    """
    df = df.copy()
    if winner_col in df.columns:
        df[winner_col] = (
            pd.to_numeric(df[winner_col], errors="coerce").fillna(0).astype(int)
        )
    return df