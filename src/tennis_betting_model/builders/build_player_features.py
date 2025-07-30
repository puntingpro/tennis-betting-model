# src/tennis_betting_model/builders/build_player_features.py

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, Tuple

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.constants import DEFAULT_PLAYER_RANK, ELO_INITIAL_RATING
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
)
from tennis_betting_model.utils.common import get_surface
from tennis_betting_model.utils.schema import validate_data


def _load_data(
    paths: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all necessary dataframes for feature building."""
    log_info("Loading consolidated data and other required files...")
    try:
        df_matches_raw = pd.read_csv(paths["betfair_match_log"])
        df_rankings = pd.read_csv(paths["consolidated_rankings"])
        df_players = pd.read_csv(paths["raw_players"], encoding="latin-1")
        df_elo = pd.read_csv(paths["elo_ratings"])

        df_players["player_id"] = pd.to_numeric(
            df_players["player_id"], errors="coerce"
        )
        df_players.dropna(subset=["player_id"], inplace=True)
        df_players["player_id"] = df_players["player_id"].astype("int64")

        return df_matches_raw, df_rankings, df_players, df_elo
    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
        raise


def calculate_player_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    log_info("Calculating player stats with anti-leakage logic...")

    if df_matches.empty:
        log_info(
            "No matches to process for player stats. Returning empty stats DataFrame."
        )
        return pd.DataFrame(
            columns=[
                "match_id",
                "player_id",
                "win_perc",
                "surface_win_perc",
                "matches_last_7_days",
                "matches_last_14_days",
            ]
        )

    player_stats_cache: DefaultDict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "matches_played": 0,
            "wins": 0,
            "surface_matches": defaultdict(int),
            "surface_wins": defaultdict(int),
            "recent_match_dates": deque(),
        }
    )
    match_features = []

    df_matches["tourney_date"] = pd.to_datetime(df_matches["tourney_date"])

    for row in tqdm(
        df_matches.itertuples(), total=len(df_matches), desc="Processing Player Stats"
    ):
        winner_id, loser_id, surface, match_date = (
            row.winner_historical_id,
            row.loser_historical_id,
            row.surface,
            row.tourney_date,
        )
        if pd.isna(winner_id) or pd.isna(loser_id):
            continue

        for player_id in [winner_id, loser_id]:
            stats = player_stats_cache[int(player_id)]

            while (
                stats["recent_match_dates"]
                and (match_date - stats["recent_match_dates"][0]).days > 14
            ):
                stats["recent_match_dates"].popleft()

            matches_last_14_days = len(stats["recent_match_dates"])
            matches_last_7_days = sum(
                1 for d in stats["recent_match_dates"] if (match_date - d).days <= 7
            )

            match_features.append(
                {
                    "match_id": row.match_id,
                    "player_id": int(player_id),
                    "win_perc": (stats["wins"] / stats["matches_played"])
                    if stats["matches_played"] > 0
                    else 0,
                    "surface_win_perc": (
                        stats["surface_wins"][surface]
                        / stats["surface_matches"][surface]
                    )
                    if stats["surface_matches"][surface] > 0
                    else 0,
                    "matches_last_7_days": matches_last_7_days,
                    "matches_last_14_days": matches_last_14_days,
                }
            )

        winner_stats = player_stats_cache[int(winner_id)]
        loser_stats = player_stats_cache[int(loser_id)]

        winner_stats["matches_played"] += 1
        winner_stats["wins"] += 1
        winner_stats["surface_matches"][surface] += 1
        winner_stats["surface_wins"][surface] += 1
        winner_stats["recent_match_dates"].append(match_date)

        loser_stats["matches_played"] += 1
        loser_stats["surface_matches"][surface] += 1
        loser_stats["recent_match_dates"].append(match_date)

    return pd.DataFrame(match_features)


def _merge_point_in_time_ranks(
    features_df: pd.DataFrame, df_rankings: pd.DataFrame
) -> pd.DataFrame:
    log_info("Calculating historical ranks for each player...")

    if features_df.empty:
        features_df["winner_rank"] = pd.Series(dtype="float64")
        features_df["loser_rank"] = pd.Series(dtype="float64")
        return features_df

    df_rankings["ranking_date"] = pd.to_datetime(df_rankings["ranking_date"])
    df_rankings = df_rankings.sort_values("ranking_date").set_index("ranking_date")

    features_df["winner_historical_id"] = features_df["winner_historical_id"].astype(
        "int64"
    )
    features_df["loser_historical_id"] = features_df["loser_historical_id"].astype(
        "int64"
    )

    winner_ranks = pd.merge_asof(
        features_df[["tourney_date", "winner_historical_id"]],
        df_rankings,
        left_on="tourney_date",
        right_index=True,
        left_by="winner_historical_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "winner_rank"})
    loser_ranks = pd.merge_asof(
        features_df[["tourney_date", "loser_historical_id"]],
        df_rankings,
        left_on="tourney_date",
        right_index=True,
        left_by="loser_historical_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "loser_rank"})
    features_df["winner_rank"] = winner_ranks["winner_rank"].fillna(DEFAULT_PLAYER_RANK)
    features_df["loser_rank"] = loser_ranks["loser_rank"].fillna(DEFAULT_PLAYER_RANK)
    return features_df


def _merge_stats_and_elo(
    features_df: pd.DataFrame, df_elo: pd.DataFrame, player_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges Elo ratings and calculated player stats into the features DataFrame."""
    # Enforce consistent dtypes on merge keys
    features_df["match_id"] = features_df["match_id"].astype(str)
    df_elo["match_id"] = df_elo["match_id"].astype(str)
    player_stats_df["match_id"] = player_stats_df["match_id"].astype(str)
    player_stats_df["player_id"] = player_stats_df["player_id"].astype("int64")

    # Merge Elo
    merged_df = pd.merge(
        features_df, df_elo, on=["match_id", "p1_id", "p2_id"], how="left"
    )

    # Merge stats for Player 1
    p1_stats = player_stats_df.rename(
        columns={
            c: f"p1_{c}"
            for c in player_stats_df.columns
            if c not in ["match_id", "player_id"]
        }
    )
    merged_df = pd.merge(
        merged_df,
        p1_stats,
        left_on=["match_id", "p1_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    # Merge stats for Player 2
    p2_stats = player_stats_df.rename(
        columns={
            c: f"p2_{c}"
            for c in player_stats_df.columns
            if c not in ["match_id", "player_id"]
        }
    )
    merged_df = pd.merge(
        merged_df,
        p2_stats,
        left_on=["match_id", "p2_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    # Drop helper player_id columns from merges
    merged_df.drop(
        columns=["player_id_x", "player_id_y"], inplace=True, errors="ignore"
    )

    return merged_df


def _add_player_attributes(
    features_df: pd.DataFrame, df_players: pd.DataFrame
) -> pd.DataFrame:
    """Adds player attributes like hand to the features DataFrame."""
    player_info = (
        df_players[["player_id", "hand"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
    )

    merged_df = features_df.merge(
        player_info, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"hand": "p1_hand"})
    merged_df = merged_df.merge(
        player_info, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"hand": "p2_hand"})

    return merged_df


def _calculate_differential_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rank, Elo, and fatigue difference features."""
    df = features_df.copy()
    df["rank_diff"] = df["p1_rank"] - df["p2_rank"]
    df["elo_diff"] = pd.to_numeric(df["p1_elo"], errors="coerce") - pd.to_numeric(
        df["p2_elo"], errors="coerce"
    )
    df["fatigue_diff_7_days"] = (
        df["p1_matches_last_7_days"] - df["p2_matches_last_7_days"]
    )
    df["fatigue_diff_14_days"] = (
        df["p1_matches_last_14_days"] - df["p2_matches_last_14_days"]
    )
    return df


def _assemble_final_features(
    features_df: pd.DataFrame,
    df_elo: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    df_players: pd.DataFrame,
) -> pd.DataFrame:
    """Assembles the complete feature set from all component dataframes."""
    # Define p1/p2 and winner columns
    features_df["p1_id"] = np.minimum(
        features_df["winner_historical_id"], features_df["loser_historical_id"]
    ).astype("int64")
    features_df["p2_id"] = np.maximum(
        features_df["winner_historical_id"], features_df["loser_historical_id"]
    ).astype("int64")
    features_df = features_df[features_df["p1_id"] != features_df["p2_id"]].copy()
    features_df["winner"] = (
        features_df["p1_id"] == features_df["winner_historical_id"]
    ).astype(int)

    p1_is_winner = features_df["p1_id"] == features_df["winner_historical_id"]
    features_df["p1_rank"] = np.where(
        p1_is_winner, features_df["winner_rank"], features_df["loser_rank"]
    )
    features_df["p2_rank"] = np.where(
        p1_is_winner, features_df["loser_rank"], features_df["winner_rank"]
    )

    # Assemble features using modular functions
    features_df = _merge_stats_and_elo(features_df, df_elo, player_stats_df)
    features_df = _add_player_attributes(features_df, df_players)

    fill_values = {
        "p1_elo": ELO_INITIAL_RATING,
        "p2_elo": ELO_INITIAL_RATING,
        "p1_win_perc": 0.0,
        "p2_win_perc": 0.0,
        "p1_surface_win_perc": 0.0,
        "p2_surface_win_perc": 0.0,
        "p1_matches_last_7_days": 0.0,
        "p2_matches_last_7_days": 0.0,
        "p1_matches_last_14_days": 0.0,
        "p2_matches_last_14_days": 0.0,
    }
    features_df.fillna(value=fill_values, inplace=True)

    features_df = _calculate_differential_features(features_df)

    # --- FIX: Drop non-feature columns that cause model training to fail ---
    cols_to_drop = [
        "winner_id",
        "winner_name",
        "winner_historical_id",
        "winner_rank",
        "loser_id",
        "loser_name",
        "loser_historical_id",
        "loser_rank",
    ]
    features_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return features_df


def main(args):
    """Main workflow for building all player features."""
    setup_logging()

    config = load_config(args.config)
    paths = config["data_paths"]

    df_matches_raw, df_rankings, df_players, df_elo = _load_data(paths)

    df_matches_raw["winner_historical_id"] = pd.to_numeric(
        df_matches_raw["winner_historical_id"], errors="coerce"
    )
    df_matches_raw["loser_historical_id"] = pd.to_numeric(
        df_matches_raw["loser_historical_id"], errors="coerce"
    )
    df_matches = validate_data(
        df_matches_raw, "betfair_match_log", "Initial Betfair Match Log"
    )

    log_info("--- Starting Feature Engineering ---")
    df_matches.dropna(
        subset=["tourney_date", "winner_historical_id", "loser_historical_id"],
        inplace=True,
    )

    if not df_matches.empty:
        df_matches = df_matches.astype(
            {"winner_historical_id": "int64", "loser_historical_id": "int64"}
        )

    features_df = df_matches.copy()
    features_df["surface"] = features_df["tourney_name"].apply(get_surface)
    log_info("Derived 'surface' from tournament names.")

    features_df["tourney_date"] = pd.to_datetime(features_df["tourney_date"])
    features_df = features_df.sort_values("tourney_date")
    features_df = _merge_point_in_time_ranks(features_df, df_rankings)

    player_stats_df = calculate_player_stats(features_df)

    final_df = _assemble_final_features(
        features_df, df_elo, player_stats_df, df_players
    )

    validated_features = validate_data(final_df, "final_features", "Final Feature Set")

    output_path = Path(paths["consolidated_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL features to {output_path}...")
    validated_features.to_csv(output_path, index=False)
    log_success(f"✅ Successfully created FINAL feature library at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)
