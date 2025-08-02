# src/tennis_betting_model/builders/build_player_features.py

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
)
from tennis_betting_model.utils.common import get_surface
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.builders.feature_builder import FeatureBuilder
from tennis_betting_model.utils.constants import ELO_INITIAL_RATING


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

        df_rankings["ranking_date"] = pd.to_datetime(
            df_rankings["ranking_date"], utc=True
        )
        df_rankings = df_rankings.sort_values("ranking_date")

        df_matches_raw["tourney_date"] = pd.to_datetime(
            df_matches_raw["tourney_date"], utc=True
        )

        df_matches_raw["winner_historical_id"] = pd.to_numeric(
            df_matches_raw["winner_historical_id"], errors="coerce"
        )
        df_matches_raw["loser_historical_id"] = pd.to_numeric(
            df_matches_raw["loser_historical_id"], errors="coerce"
        )
        df_matches_raw.dropna(
            subset=["winner_historical_id", "loser_historical_id"], inplace=True
        )
        df_matches_raw = df_matches_raw.astype(
            {"winner_historical_id": "int64", "loser_historical_id": "int64"}
        )
        df_matches_raw["match_id"] = df_matches_raw["match_id"].astype(str)
        df_elo["match_id"] = df_elo["match_id"].astype(str)

        df_matches_raw["surface"] = df_matches_raw["tourney_name"].apply(get_surface)

        return df_matches_raw, df_rankings, df_players, df_elo
    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
        raise


def main(args):
    """Main workflow for building all player features using the unified FeatureBuilder."""
    setup_logging()

    config = load_config(args.config)
    paths = config["data_paths"]

    df_matches, df_rankings, df_players, df_elo = _load_data(paths)

    df_matches = validate_data(
        df_matches, "betfair_match_log", "Initial Betfair Match Log"
    )

    log_info("--- Starting Feature Engineering using Unified FeatureBuilder ---")

    player_info_lookup = (
        df_players[["player_id", "hand"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
        .to_dict("index")
    )

    feature_builder = FeatureBuilder(
        player_info_lookup=player_info_lookup,
        df_rankings=df_rankings,
        df_matches=df_matches,
        df_elo=df_elo,
    )

    all_features = []

    df_matches = df_matches.sort_values("tourney_date").reset_index(drop=True)

    for row in tqdm(
        df_matches.itertuples(),
        total=len(df_matches),
        desc="Building Historical Features",
    ):
        p1_id = min(row.winner_historical_id, row.loser_historical_id)
        p2_id = max(row.winner_historical_id, row.loser_historical_id)

        if p1_id == p2_id:
            continue

        surface = row.surface

        feature_dict = feature_builder.build_features(
            p1_id=p1_id,
            p2_id=p2_id,
            surface=surface,
            match_date=row.tourney_date,
            match_id=row.match_id,
        )

        feature_dict["match_id"] = row.match_id
        feature_dict["tourney_date"] = row.tourney_date
        feature_dict["tourney_name"] = row.tourney_name
        feature_dict["surface"] = surface
        feature_dict["winner"] = 1 if p1_id == row.winner_historical_id else 0

        all_features.append(feature_dict)

    final_df = pd.DataFrame(all_features)

    if final_df.empty:
        log_error("No features were generated. The resulting DataFrame is empty.")
        return

    elo_cols = ["p1_elo", "p2_elo", "elo_diff"]
    for col in elo_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
        # --- FIX: Use direct assignment to fill NaN values and avoid FutureWarning ---
        final_df[col] = final_df[col].fillna(ELO_INITIAL_RATING)

    validated_features = validate_data(final_df, "final_features", "Final Feature Set")

    output_path = Path(paths["consolidated_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL features to {output_path}...")
    validated_features.to_csv(output_path, index=False)
    log_success(f"✅ Successfully created FINAL feature library at {output_path}")
