# src/scripts/utils/common.py
import numpy as np
import pandas as pd


def get_most_recent_ranking(
    df_rankings: pd.DataFrame, player_id: int, date: pd.Timestamp
) -> float:
    """
    Retrieves the most recent ranking for a player before a given date.

    Args:
        df_rankings (pd.DataFrame): DataFrame of historical rankings.
        player_id (int): The ID of the player.
        date (pd.Timestamp): The date to find the most recent ranking before.

    Returns:
        float: The most recent rank, or np.nan if not found.
    """
    player_rankings = df_rankings[df_rankings["player"] == player_id]
    last_ranking_idx = (
        player_rankings["ranking_date"].searchsorted(date, side="right") - 1
    )
    if last_ranking_idx >= 0:
        return float(player_rankings.iloc[last_ranking_idx]["rank"])
    return np.nan


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names to lowercase and replaces spaces with underscores.

    Args:
        df (pd.DataFrame): The DataFrame to normalize.

    Returns:
        pd.DataFrame: The DataFrame with normalized column names.
    """
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def patch_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a numeric 'winner' column exists in the DataFrame.

    It checks for a 'result' column and converts it if 'winner' is missing,
    and ensures the 'winner' column is numeric if it already exists.

    Args:
        df (pd.DataFrame): The DataFrame to patch.

    Returns:
        pd.DataFrame: The DataFrame with a guaranteed numeric 'winner' column.
    """
    if "winner" not in df.columns and "result" in df.columns:
        df["winner"] = pd.to_numeric(df["result"], errors="coerce")
    elif "winner" in df.columns:
        df["winner"] = pd.to_numeric(df["winner"], errors="coerce")

    return df
