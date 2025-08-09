# src/tennis_betting_model/analysis/list_tournaments.py

import pandas as pd
from pathlib import Path
from typing import Optional

from src.tennis_betting_model.utils.config_schema import Config
from src.tennis_betting_model.utils.logger import setup_logging, log_info, log_error


def main_cli(config: Config, year: Optional[int] = None) -> None:
    """
    Main CLI entrypoint for listing unique tournament names from the match log.
    """
    setup_logging()
    paths = config.data_paths

    try:
        match_log_path = Path(paths.betfair_match_log)
        log_info(f"Loading match log from {match_log_path}...")
        df = pd.read_csv(match_log_path, parse_dates=["tourney_date"])

        if year:
            log_info(f"Filtering for year: {year}")
            df = df[df["tourney_date"].dt.year == year]

        if df.empty:
            log_error("No tournament data found for the specified criteria.")
            return

        unique_tournaments = sorted([str(name) for name in df["tourney_name"].unique()])

        log_info(f"Found {len(unique_tournaments)} unique tournaments:")
        print("---" * 10)
        for name in unique_tournaments:
            print(name)
        print("---" * 10)

    except FileNotFoundError:
        log_error(f"Match log not found at {match_log_path}.")
        log_error("Please run the 'build' command first to generate the match log.")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
