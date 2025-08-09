# src/tennis_betting_model/builders/build_player_features.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.tennis_betting_model.utils.config_schema import Config
from src.tennis_betting_model.utils.data_loader import DataLoader
from src.tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
)
from src.tennis_betting_model.utils.schema import validate_data
from src.tennis_betting_model.builders.feature_logic import get_h2h_stats_optimized
from src.tennis_betting_model.builders.vectorized_features import (
    build_vectorized_features,
)
from src.tennis_betting_model.utils.common import get_most_recent_ranking


def main(config: Config):
    """Main workflow for building all player features using a high-performance, vectorized approach."""
    setup_logging()

    data_loader = DataLoader(config.data_paths)

    try:
        (
            df_matches,
            df_rankings,
            df_players,
            df_elo,
            player_info_lookup,
        ) = data_loader.load_all_pipeline_data()

        log_info(
            f"Loading backtest market data from {config.data_paths.backtest_market_data}..."
        )
        df_market_data = pd.read_csv(
            config.data_paths.backtest_market_data, dtype={"match_id": str}
        )

        # Add player IDs to the main matches df to ensure correct player alignment (p1_id < p2_id)
        df_matches = pd.merge(
            df_matches,
            df_market_data[["match_id", "p1_id", "p2_id"]],
            on="match_id",
            how="inner",
        )

    except FileNotFoundError:
        log_error(
            "A required data file was not found. Please run 'prepare-data' first."
        )
        return

    log_info("--- Starting High-Performance Vectorized Feature Engineering ---")

    # 1. Build core player-specific features using the new vectorized function
    df_features = build_vectorized_features(df_matches)

    # 2. Merge match-specific features
    log_info("Merging Elo ratings...")
    df_features = df_features.merge(
        df_elo[["match_id", "p1_elo", "p2_elo"]], on="match_id", how="left"
    )

    df_features["p1_elo"].fillna(config.elo_config.initial_rating, inplace=True)
    df_features["p2_elo"].fillna(config.elo_config.initial_rating, inplace=True)

    df_features["elo_diff"] = df_features["p1_elo"] - df_features["p2_elo"]

    log_info("Calculating Head-to-Head stats...")
    h2h_df = df_matches.copy()
    h2h_df["p1_id_h2h"] = h2h_df[["winner_historical_id", "loser_historical_id"]].min(
        axis=1
    )
    h2h_df["p2_id_h2h"] = h2h_df[["winner_historical_id", "loser_historical_id"]].max(
        axis=1
    )
    h2h_df = h2h_df.set_index(["p1_id_h2h", "p2_id_h2h"]).sort_index()

    tqdm.pandas(desc="Calculating H2H Stats")
    h2h_stats = df_features.progress_apply(  # type: ignore
        lambda row: get_h2h_stats_optimized(
            h2h_df, row["p1_id"], row["p2_id"], row["tourney_date"], row["surface"]
        ),
        axis=1,
    )
    df_features[["h2h_surface_p1_wins", "h2h_surface_p2_wins"]] = pd.DataFrame(
        h2h_stats.tolist(), index=h2h_stats.index
    )

    log_info("Adding final features (ranks, odds, etc.)...")

    # Correctly look up the most recent rank for each player at match time
    tqdm.pandas(desc="Looking up Player 1 Ranks")
    df_features["p1_rank"] = df_features.progress_apply(lambda row: get_most_recent_ranking(df_rankings, row["p1_id"], row["tourney_date"], config.elo_config.default_player_rank), axis=1)  # type: ignore

    tqdm.pandas(desc="Looking up Player 2 Ranks")
    df_features["p2_rank"] = df_features.progress_apply(lambda row: get_most_recent_ranking(df_rankings, row["p2_id"], row["tourney_date"], config.elo_config.default_player_rank), axis=1)  # type: ignore
    df_features["rank_diff"] = df_features["p1_rank"] - df_features["p2_rank"]

    # Merge odds info from backtest data
    df_features = df_features.merge(
        df_market_data[["match_id", "p1_odds", "p2_odds"]], on="match_id", how="left"
    )
    df_features["p1_implied_prob"] = 1 / df_features["p1_odds"]
    df_features["p2_implied_prob"] = 1 / df_features["p2_odds"]
    df_features["book_margin"] = (
        df_features["p1_implied_prob"] + df_features["p2_implied_prob"]
    ) - 1

    for col in ["p1_implied_prob", "p2_implied_prob", "book_margin"]:
        df_features[col].fillna(0, inplace=True)

    # Add winner column for model training
    df_features["winner"] = (
        df_features["winner_historical_id"] == df_features["p1_id"]
    ).astype(int)

    # Rename columns to their final schema names
    df_features.rename(
        columns={"p1_form_10": "p1_form", "p2_form_10": "p2_form"}, inplace=True
    )

    # Fatigue diff features
    df_features["fatigue_diff_7_days"] = (
        df_features["p1_matches_last_7_days"] - df_features["p2_matches_last_7_days"]
    )
    df_features["fatigue_diff_14_days"] = (
        df_features["p1_matches_last_14_days"] - df_features["p2_matches_last_14_days"]
    )
    df_features["fatigue_sets_diff_7_days"] = (
        df_features["p1_sets_played_last_7_days"]
        - df_features["p2_sets_played_last_7_days"]
    )
    df_features["fatigue_sets_diff_14_days"] = (
        df_features["p1_sets_played_last_14_days"]
        - df_features["p2_sets_played_last_14_days"]
    )

    # Merge hand info and clean up merge artifacts ('player_id_x', 'player_id_y')
    df_features = (
        df_features.merge(
            df_players[["player_id", "hand"]],
            left_on="p1_id",
            right_on="player_id",
            how="left",
        )
        .rename(columns={"hand": "p1_hand"})
        .drop(columns=["player_id"])
    )

    df_features = (
        df_features.merge(
            df_players[["player_id", "hand"]],
            left_on="p2_id",
            right_on="player_id",
            how="left",
        )
        .rename(columns={"hand": "p2_hand"})
        .drop(columns=["player_id"])
    )

    final_df = df_features.rename(columns={"match_id": "market_id"})

    validated_features = validate_data(final_df, "final_features", "Final Feature Set")

    output_path = Path(config.data_paths.consolidated_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL features to {output_path}...")
    validated_features.to_csv(output_path, index=False)
    log_success(f"✅ Successfully created FINAL feature library at {output_path}")
