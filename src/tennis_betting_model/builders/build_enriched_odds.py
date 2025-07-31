# src/tennis_betting_model/builders/build_enriched_odds.py
import pandas as pd
import glob
import os
from pathlib import Path
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    setup_logging,
)


def main():
    """
    Finds and consolidates all Betfair summary CSV files (*_ProTennis.csv)
    from the raw data directory into a single file.
    """
    setup_logging()
    config = load_config("config.yaml")
    paths = config["data_paths"]
    raw_data_path = paths["raw_data_dir"]
    output_path = Path(paths["betfair_raw_odds"])

    log_info(f"Searching for summary files in {raw_data_path}...")
    summary_files = glob.glob(os.path.join(raw_data_path, "*_ProTennis.csv"))

    if not summary_files:
        log_error(
            f"No summary CSV files found in {raw_data_path}. Please download them first."
        )
        return

    log_info(f"Found {len(summary_files)} summary files to consolidate.")
    df_list = [pd.read_csv(f) for f in summary_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    if "event_date" not in combined_df.columns:
        log_error(
            "Critical error: 'event_date' column not found in the source summary CSV file."
        )
        return

    # --- FIX: Convert tourney_date to UTC and handle DD/MM/YYYY format ---
    combined_df["tourney_date"] = pd.to_datetime(
        combined_df["event_date"], errors="coerce", dayfirst=True, utc=True
    )

    combined_df.dropna(
        subset=["tourney_date", "market_id", "selection_id"], inplace=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    log_success(
        f"Successfully consolidated {len(combined_df)} records into {output_path}"
    )


if __name__ == "__main__":
    main()
