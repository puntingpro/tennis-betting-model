# src/tennis_betting_model/builders/build_player_features.py

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Tuple

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.constants import DEFAULT_PLAYER_RANK
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
        return df_matches_raw, df_rankings, df_players, df_elo
    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
        raise


def calculate_player_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Calculates historical player stats with anti-leakage logic."""
    log_info("Calculating player stats with anti-leakage logic...")
    player_stats_cache: DefaultDict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "matches_played": 0,
            "wins": 0,
            "surface_matches": defaultdict(int),
            "surface_wins": defaultdict(int),
        }
    )
    match_features = []

    for row in tqdm(
        df_matches.itertuples(), total=len(df_matches), desc="Processing Player Stats"
    ):
        winner_id, loser_id, surface = (
            row.winner_historical_id,
            row.loser_historical_id,
            row.surface,
        )

        if pd.isna(winner_id) or pd.isna(loser_id):
            continue

        winner_stats = player_stats_cache[int(winner_id)]
        loser_stats = player_stats_cache[int(loser_id)]

        match_features.append(
            {
                "match_id": row.match_id,
                "player_id": int(winner_id),
                "win_perc": (winner_stats["wins"] / winner_stats["matches_played"])
                if winner_stats["matches_played"] > 0
                else 0,
                "surface_win_perc": (
                    winner_stats["surface_wins"][surface]
                    / winner_stats["surface_matches"][surface]
                )
                if winner_stats["surface_matches"][surface] > 0
                else 0,
            }
        )
        match_features.append(
            {
                "match_id": row.match_id,
                "player_id": int(loser_id),
                "win_perc": (loser_stats["wins"] / loser_stats["matches_played"])
                if loser_stats["matches_played"] > 0
                else 0,
                "surface_win_perc": (
                    loser_stats["surface_wins"][surface]
                    / loser_stats["surface_matches"][surface]
                )
                if loser_stats["surface_matches"][surface] > 0
                else 0,
            }
        )

        winner_stats["matches_played"] += 1
        winner_stats["wins"] += 1
        winner_stats["surface_matches"][surface] += 1
        winner_stats["surface_wins"][surface] += 1

        loser_stats["matches_played"] += 1
        loser_stats["surface_matches"][surface] += 1

    return pd.DataFrame(match_features)


def _merge_point_in_time_ranks(
    features_df: pd.DataFrame, df_rankings: pd.DataFrame
) -> pd.DataFrame:
    """Merges historical ranks onto the features dataframe for each player."""
    log_info("Calculating historical ranks for each player...")
    df_rankings["ranking_date"] = pd.to_datetime(df_rankings["ranking_date"])
    df_rankings = df_rankings.sort_values("ranking_date").set_index("ranking_date")

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


def _assemble_final_features(
    features_df: pd.DataFrame,
    df_elo: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    df_players: pd.DataFrame,
) -> pd.DataFrame:
    """Assembles all calculated features into a final dataframe."""
    log_info("Assembling final feature set...")
    features_df["p1_id"] = np.minimum(
        features_df["winner_historical_id"], features_df["loser_historical_id"]
    )
    features_df["p2_id"] = np.maximum(
        features_df["winner_historical_id"], features_df["loser_historical_id"]
    )
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

    features_df = pd.merge(
        features_df, df_elo, on=["match_id", "p1_id", "p2_id"], how="left"
    )

    p1_stats = player_stats_df.rename(
        columns=lambda c: f"p1_{c}" if c not in ["match_id", "player_id"] else c
    )
    features_df = pd.merge(
        features_df,
        p1_stats,
        left_on=["match_id", "p1_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    p2_stats = player_stats_df.rename(
        columns=lambda c: f"p2_{c}" if c not in ["match_id", "player_id"] else c
    )
    features_df = pd.merge(
        features_df,
        p2_stats,
        left_on=["match_id", "p2_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    player_info = (
        df_players[["player_id", "hand", "height"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
    )
    features_df = features_df.merge(
        player_info, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"hand": "p1_hand", "height": "p1_height"})
    features_df = features_df.merge(
        player_info, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"hand": "p2_hand", "height": "p2_height"})

    features_df["rank_diff"] = features_df["p1_rank"] - features_df["p2_rank"]
    features_df["elo_diff"] = features_df["p1_elo"] - features_df["p2_elo"]

    final_columns = [
        "match_id",
        "tourney_date",
        "tourney_name",
        "surface",
        "p1_id",
        "p2_id",
        "winner",
        "p1_rank",
        "p2_rank",
        "rank_diff",
        "p1_elo",
        "p2_elo",
        "elo_diff",
        "p1_win_perc",
        "p2_win_perc",
        "p1_surface_win_perc",
        "p2_surface_win_perc",
        "p1_hand",
        "p2_hand",
        "p1_height",
        "p2_height",
    ]

    existing_cols = [col for col in final_columns if col in features_df.columns]
    return features_df[existing_cols]


def main(args):
    """Main workflow for building all player features."""
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]

    df_matches_raw, df_rankings, df_players, df_elo = _load_data(paths)

    df_matches = validate_data(
        df_matches_raw, "betfair_match_log", "Initial Betfair Match Log"
    )
    log_info("--- Starting Feature Engineering ---")

    df_matches["winner_historical_id"] = pd.to_numeric(
        df_matches["winner_historical_id"], errors="coerce"
    )
    df_matches["loser_historical_id"] = pd.to_numeric(
        df_matches["loser_historical_id"], errors="coerce"
    )
    df_matches.dropna(
        subset=["tourney_date", "winner_historical_id", "loser_historical_id"],
        inplace=True,
    )
    df_matches = df_matches.astype(
        {"winner_historical_id": "int64", "loser_historical_id": "int64"}
    )

    features_df = df_matches.copy()
    features_df["surface"] = features_df["tourney_name"].apply(get_surface)
    log_info("Derived 'surface' from tournament names.")

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
