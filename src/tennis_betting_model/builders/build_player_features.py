# src/tennis_betting_model/builders/build_player_features.py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

from tennis_betting_model.utils.config import load_config, Config
from tennis_betting_model.utils.data_loader import DataLoader
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
)
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.builders.feature_builder import FeatureBuilder


def main(args):
    """Main workflow for building all player features using a stateful, chronological approach."""
    setup_logging()
    config = Config(**load_config(args.config))
    data_loader = DataLoader(config.data_paths)

    try:
        (
            df_matches,
            df_rankings,
            _,
            df_elo,
            player_info_lookup,
        ) = data_loader.load_all_pipeline_data()

        log_info(
            f"Loading backtest market data from {config.data_paths.backtest_market_data}..."
        )
        df_market_data = pd.read_csv(config.data_paths.backtest_market_data)
        df_market_data["match_id"] = df_market_data["market_id"].astype(str)

        # --- FIX: Include 'p1_id' in the merge and update fillna syntax ---
        df_matches = pd.merge(
            df_matches,
            df_market_data[["match_id", "p1_id", "p1_odds", "p2_odds"]],
            on="match_id",
            how="left",
        )
        df_matches["p1_odds"] = df_matches["p1_odds"].fillna(0)
        df_matches["p2_odds"] = df_matches["p2_odds"].fillna(0)

    except FileNotFoundError:
        log_error(
            "A required data file was not found. Please run 'prepare-data' and create the necessary files."
        )
        return

    log_info("--- Starting High-Performance Feature Engineering ---")
    df_matches = df_matches.sort_values("tourney_date").reset_index(drop=True)

    feature_builder = FeatureBuilder(
        player_info_lookup=player_info_lookup,
        df_rankings=df_rankings,
        df_matches=df_matches,
        df_elo=df_elo,
        elo_config=config.elo_config,
    )

    all_features: List[Dict[str, Any]] = []

    log_info(f"Generating features for {len(df_matches)} historical matches...")
    for row in tqdm(
        df_matches.itertuples(),
        total=len(df_matches),
        desc="Building Historical Features",
    ):
        p1_id = min(row.winner_historical_id, row.loser_historical_id)
        p2_id = max(row.winner_historical_id, row.loser_historical_id)

        if p1_id == p2_id:
            continue

        if row.p1_id == p1_id:
            p1_odds = row.p1_odds
            p2_odds = row.p2_odds
        else:
            p1_odds = row.p2_odds
            p2_odds = row.p1_odds

        features = feature_builder.build_features(
            p1_id=p1_id,
            p2_id=p2_id,
            surface=row.surface,
            match_date=row.tourney_date,
            match_id=row.match_id,
            p1_odds=p1_odds,
            p2_odds=p2_odds,
        )

        features["winner"] = 1 if p1_id == row.winner_historical_id else 0
        features["tourney_name"] = row.tourney_name
        features["tourney_date"] = row.tourney_date
        features["surface"] = row.surface

        all_features.append(features)

    if not all_features:
        log_error("No features were generated. The resulting DataFrame is empty.")
        return

    final_df = pd.DataFrame(all_features)

    elo_cols = ["p1_elo", "p2_elo", "elo_diff"]
    for col in elo_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
        final_df[col] = final_df[col].fillna(config.elo_config.initial_rating)

    validated_features = validate_data(final_df, "final_features", "Final Feature Set")

    output_path = Path(config.data_paths.consolidated_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL features to {output_path}...")
    validated_features.to_csv(output_path, index=False)
    log_success(f"✅ Successfully created FINAL feature library at {output_path}")
