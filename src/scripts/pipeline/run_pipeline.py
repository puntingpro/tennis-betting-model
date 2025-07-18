# src/scripts/pipeline/run_pipeline.py

import pandas as pd
import joblib
from src.scripts.utils.logger import setup_logging, log_info, log_warning, log_success
from src.scripts.utils.config import load_config
from src.scripts.utils.api import (
    login_to_betfair,
    get_tennis_competitions,
    get_live_match_odds,
    place_bet,
)
from src.scripts.utils.data_loader import load_pipeline_data
from src.scripts.pipeline.value_finder import process_markets


def run_pipeline_once(config: dict, dry_run: bool):
    """
    Runs a single iteration of the value-finding pipeline.
    """
    paths = config["data_paths"]
    betting_config = config["betting"]

    if dry_run:
        log_warning("🚀 Running in DRY-RUN mode. No real bets will be placed.")
    else:
        log_info("🚀 Running in LIVE mode.")

    # In a real application, you would fetch your live bankroll here.
    # For now, we will use a placeholder value from the config.
    bankroll = float(betting_config.get("live_bankroll", 1000.0))
    log_info(f"Using bankroll: ${bankroll:,.2f}")

    model = joblib.load(paths["model"])
    player_info_lookup, df_rankings, df_matches = load_pipeline_data(paths)

    trading = login_to_betfair(config)
    try:
        target_competition_ids = get_tennis_competitions(
            trading, betting_config["profitable_tournaments"]
        )
        if not target_competition_ids:
            log_info(
                "No live competitions found matching profitable tournament keywords. Exiting."
            )
            return

        log_info(f"Found {len(target_competition_ids)} target competitions to scan.")
        market_catalogues, market_book_lookup = get_live_match_odds(
            trading, target_competition_ids
        )
        if not market_catalogues:
            log_info(
                "No live matches found in targeted competitions at this time. Exiting."
            )
            return

        value_bets = process_markets(
            model,
            market_catalogues,
            market_book_lookup,
            player_info_lookup,
            df_rankings,
            df_matches,
            betting_config,
        )

        if not value_bets:
            log_info("--- No Value Bets Found in This Run ---")
            return

        if dry_run:
            log_info("Value bets found and alerted in DRY-RUN mode.")
            return

        log_warning(
            f"Found {len(value_bets)} value bets. Attempting to place live bets..."
        )
        for bet in value_bets:
            # Calculate stake using the Kelly Criterion fraction from the value_finder
            # You might want to add your own logic here to cap the kelly fraction
            kelly_fraction = float(bet.get("kelly_fraction", 0.0))
            stake_to_place = bankroll * kelly_fraction

            place_bet(
                trading=trading,
                market_id=bet["market_id"],
                selection_id=int(bet["selection_id"]),
                price=float(bet["odds"]),
                stake=stake_to_place,
            )

    finally:
        trading.logout()
        log_info("\nLogged out.")


def main(args):
    """
    Main CLI handler for running a single pipeline instance.
    """
    setup_logging()
    config = load_config(args.config)
    run_pipeline_once(config, args.dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without placing bets.",
    )
    args = parser.parse_args()
    main(args)
