# src/tennis_betting_model/builders/build_match_log.py
import pandas as pd
from pathlib import Path
from src.tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    log_warning,
)


def main(config: dict):
    """
    Creates the historical match log from the consolidated summary file.
    """
    log_info("--- Creating Match Log from Summary File ---")
    paths = config["data_paths"]

    log_info("Loading consolidated raw odds and player map...")
    raw_odds_path = Path(paths["betfair_raw_odds"])
    map_path = Path(paths["player_map"])

    if not raw_odds_path.exists() or not map_path.exists():
        log_error(
            "Raw odds or player map file not found. Run 'prepare-data' and 'create-player-map' first."
        )
        return

    # --- FIX: Ensure date column is parsed as timezone-aware ---
    df_raw = pd.read_csv(raw_odds_path, parse_dates=["tourney_date"])
    df_raw["tourney_date"] = pd.to_datetime(df_raw["tourney_date"], utc=True)

    df_map = pd.read_csv(
        map_path, dtype={"betfair_id": "int64", "historical_id": "Int64"}
    )

    log_info("Enriching summary data with historical IDs...")
    df_enriched = pd.merge(
        df_raw, df_map, left_on="selection_id", right_on="betfair_id", how="left"
    )

    df_settled = df_enriched[df_enriched["result"].isin(["WINNER", "LOSER"])].copy()

    winners = df_settled[df_settled["result"] == "WINNER"].copy()
    losers = df_settled[df_settled["result"] == "LOSER"].copy()

    winners.rename(
        columns={
            "selection_id": "winner_id",
            "selection_name": "winner_name",
            "historical_id": "winner_historical_id",
        },
        inplace=True,
    )
    losers.rename(
        columns={
            "selection_id": "loser_id",
            "selection_name": "loser_name",
            "historical_id": "loser_historical_id",
        },
        inplace=True,
    )

    key_cols = ["market_id", "tourney_date", "competition_name"]
    match_log_df = pd.merge(
        winners[key_cols + ["winner_id", "winner_name", "winner_historical_id"]],
        losers[key_cols + ["loser_id", "loser_name", "loser_historical_id"]],
        on=key_cols,
    )

    match_log_df.rename(
        columns={"competition_name": "tourney_name", "market_id": "match_id"},
        inplace=True,
    )

    if match_log_df.empty:
        log_warning(
            "⚠️ No valid, settled matches were found in the summary data. The match log will be empty."
        )
        # Save an empty file with headers to prevent downstream errors
        output_path = Path(paths["betfair_match_log"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "match_id",
                "tourney_date",
                "tourney_name",
                "winner_id",
                "winner_historical_id",
                "winner_name",
                "loser_id",
                "loser_historical_id",
                "loser_name",
            ]
        ).to_csv(output_path, index=False)
        return

    match_log_df.sort_values(by="tourney_date", inplace=True)
    output_path = Path(paths["betfair_match_log"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_cols = [
        "match_id",
        "tourney_date",
        "tourney_name",
        "winner_id",
        "winner_historical_id",
        "winner_name",
        "loser_id",
        "loser_historical_id",
        "loser_name",
    ]
    match_log_df[final_cols].to_csv(output_path, index=False)
    log_success(f"Successfully created match log with {len(match_log_df)} entries.")
