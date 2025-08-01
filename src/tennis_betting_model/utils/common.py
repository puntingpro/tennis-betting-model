# mypy: disable-error-code="no-any-return"
# src/tennis_betting_model/utils/common.py

import pandas as pd
from typing import cast
from tennis_betting_model.utils.constants import DEFAULT_PLAYER_RANK


def get_most_recent_ranking(
    df_rankings: pd.DataFrame, player_id: int, match_date: pd.Timestamp
) -> int:
    """
    Finds the most recent ranking for a player prior to a given date.
    Assumes df_rankings is sorted by ranking_date.
    """
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

    player_rankings = df_rankings[
        (df_rankings["player"] == player_id)
        & (df_rankings["ranking_date"] < match_date)
    ]

    if not player_rankings.empty:
        return cast(int, player_rankings["rank"].iloc[-1])

    return DEFAULT_PLAYER_RANK


def get_surface(tourney_name: str) -> str:
    """Determines the court surface from the tournament name."""
    name = str(tourney_name).lower()

    # --- REFINEMENT: Check for explicit surface hints in the name first ---
    if "(clay)" in name:
        return "Clay"
    if "(grass)" in name:
        return "Grass"
    if "(hard)" in name:
        return "Hard"

    # --- REFINEMENT: Use existing keyword list as a fallback ---
    clay_keywords = ["roland garros", "french open", "monte carlo", "madrid", "rome"]
    grass_keywords = [
        "wimbledon",
        "queens club",
        "halle",
        "'s-hertogenbosch",
        "newport",
    ]

    if any(keyword in name for keyword in grass_keywords):
        return "Grass"
    if any(keyword in name for keyword in clay_keywords):
        return "Clay"

    return "Hard"


def get_tournament_category(tourney_name: str) -> str:
    """
    Categorizes a tournament name into a broader category for better analysis.
    """
    tourney_name = str(tourney_name).lower()

    category_map = {
        # --- REFINEMENT: Add UTR category ---
        "utr": "UTR / Pro Series",
        "grand slam": "Grand Slam",
        "australian open": "Grand Slam",
        "roland garros": "Grand Slam",
        "french open": "Grand Slam",
        "wimbledon": "Grand Slam",
        "us open": "Grand Slam",
        "masters": "Masters 1000",
        "tour finals": "Tour Finals",
        "next gen finals": "Tour Finals",
        "atp cup": "Team Event",
        "davis cup": "Team Event",
        "laver cup": "Team Event",
        "olympics": "Olympics",
        "challenger": "Challenger",
        "chall": "Challenger",
        "itf": "ITF / Futures",
        "futures": "ITF / Futures",
    }

    for keyword, category in category_map.items():
        if keyword in tourney_name:
            return category

    return "ATP / WTA Tour"


def normalize_df_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names in a DataFrame to a standard format.
    """
    rename_dict = {
        col: col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for col in df.columns
    }
    return df.rename(columns=rename_dict)


def patch_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the 'winner' column exists and is integer type."""
    if "winner" not in df.columns:
        df["winner"] = 0
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce").fillna(0).astype(int)
    return df
