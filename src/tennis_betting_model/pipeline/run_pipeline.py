import argparse
import joblib
from tennis_betting_model.utils.logger import setup_logging, log_info, log_warning
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.api import (
    login_to_betfair,
    get_tennis_competitions,
    get_live_match_odds,
    place_bet,
)
from tennis_betting_model.utils.data_loader import load_pipeline_data
from tennis_betting_model.pipeline.value_finder import process_markets
from tennis_betting_model.utils.alerter import (
    alert_pipeline_success,
    alert_pipeline_error,
)


def run_pipeline_once(config: dict, dry_run: bool):
    """Runs a single iteration of the value-finding pipeline."""
    paths = config["data_paths"]
    betting_config = config["betting"]

    if dry_run:
        log_warning("🚀 Running in DRY-RUN mode. No real bets will be placed.")
    else:
        log_info("🚀 Running in LIVE mode.")

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
            alert_pipeline_success(bets_found=0)
            log_info(
                "No live competitions found matching profitable tournament keywords. Exiting."
            )
            return

        log_info(f"Found {len(target_competition_ids)} target competitions to scan.")
        market_catalogues, market_book_lookup = get_live_match_odds(
            trading, target_competition_ids
        )

        value_bets = process_markets(
            model,
            market_catalogues,
            market_book_lookup,
            player_info_lookup,
            df_rankings,
            df_matches,
            betting_config,
        )

        if value_bets:
            if not dry_run:
                log_warning(f"Attempting to place {len(value_bets)} live bets...")
                for bet in value_bets:
                    kelly_fraction = float(bet.get("kelly_fraction", 0.0))
                    stake_to_place = bankroll * kelly_fraction
                    place_bet(
                        trading=trading,
                        market_id=bet["market_id"],
                        selection_id=int(bet["selection_id"]),
                        price=float(bet["odds"]),
                        stake=stake_to_place,
                    )

        alert_pipeline_success(bets_found=len(value_bets))

    except Exception as e:
        alert_pipeline_error(e)
        raise

    finally:
        if trading and trading.session_token:
            trading.logout()
            log_info("\nLogged out.")


def main_cli(args):
    """Main CLI handler for running a single pipeline instance."""
    setup_logging()
    config = load_config(args.config)
    run_pipeline_once(config, args.dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without placing bets.",
    )
    cli_args = parser.parse_args()
    main_cli(cli_args)
