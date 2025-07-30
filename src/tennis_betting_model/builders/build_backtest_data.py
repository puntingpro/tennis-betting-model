# src/tennis_betting_model/builders/build_backtest_data.py
# REFACTOR: This script is now refactored to read the consolidated summary CSV file.

import pandas as pd
from pathlib import Path
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    setup_logging,
)
from tennis_betting_model.utils.config import load_config


def main():
    """
    Pre-processes the summary data file to create a clean data asset for backtesting.
    """
    setup_logging()
    log_info("--- Building Clean Backtest Market Data from Summary File ---")

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]

        raw_odds_path = Path(paths["betfair_raw_odds"])
        map_path = Path(paths["player_map"])
        output_path = Path(paths["backtest_market_data"])

        log_info(f"Loading consolidated summary data from {raw_odds_path}...")
        df_raw = pd.read_csv(raw_odds_path, parse_dates=["tourney_date"])
        df_map = pd.read_csv(
            map_path, dtype={"betfair_id": "int64", "historical_id": "Int64"}
        )

        df_enriched = pd.merge(
            df_raw, df_map, left_on="selection_id", right_on="betfair_id", how="left"
        )
        df_enriched.dropna(subset=["historical_id"], inplace=True)

        # Count runners per market to ensure we only process 2-runner markets
        market_counts = (
            df_enriched.groupby("market_id").size().reset_index(name="runner_count")
        )
        two_runner_markets = market_counts[market_counts["runner_count"] == 2]
        log_info(
            f"Found {len(two_runner_markets)} markets with exactly two fully mapped runners."
        )

        df = df_enriched[
            df_enriched["market_id"].isin(two_runner_markets["market_id"])
        ].copy()

        # Reshape the data
        p1_df = df.groupby("market_id").first().reset_index()
        p2_df = df.groupby("market_id").last().reset_index()

        market_data = pd.merge(
            p1_df[["market_id", "tourney_date", "historical_id", "pp_wap", "result"]],
            p2_df[["market_id", "historical_id", "pp_wap"]],
            on="market_id",
            suffixes=("_p1", "_p2"),
        )

        market_data.rename(
            columns={
                "historical_id_p1": "p1_id",
                "historical_id_p2": "p2_id",
                "pp_wap_p1": "p1_odds",
                "pp_wap_p2": "p2_odds",
            },
            inplace=True,
        )

        # --- REFACTOR: Determine winner using the correct column name 'result' ---
        market_data["winner"] = (market_data["result"] == "WINNER").astype(int)
        # --- END REFACTOR ---

        final_cols = [
            "market_id",
            "tourney_date",
            "p1_id",
            "p2_id",
            "p1_odds",
            "p2_odds",
            "winner",
        ]
        final_market_data = market_data[final_cols].copy()
        final_market_data["p1_id"] = final_market_data["p1_id"].astype("Int64")
        final_market_data["p2_id"] = final_market_data["p2_id"].astype("Int64")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_market_data.to_csv(output_path, index=False)
        log_success(
            f"Successfully built backtest market data with {len(final_market_data)} markets."
        )

    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
