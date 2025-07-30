import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.tennis_betting_model.builders.data_preparer import (
    consolidate_player_attributes,
    consolidate_rankings,
)
from src.tennis_betting_model.builders.player_mapper import create_mapping_file
from src.tennis_betting_model.builders.build_match_log import main as build_match_log
from src.tennis_betting_model.builders.build_elo_ratings import main as build_elo
from src.tennis_betting_model.builders.build_player_features import (
    main as build_features,
)
from src.tennis_betting_model.builders.build_enriched_odds import main as build_raw_odds
from src.tennis_betting_model.builders.build_backtest_data import (
    main as build_backtest_data,
)
from src.tennis_betting_model.modeling.train_eval_model import main_cli as train_model
from src.tennis_betting_model.pipeline.run_pipeline import main_cli as run_pipeline
from src.tennis_betting_model.analysis.run_backtest import main as run_backtest
from src.tennis_betting_model.analysis.summarize_value_bets_by_tournament import (
    main_cli as summarize_tournaments,
)
from src.tennis_betting_model.analysis.analyze_profitability import (
    main_cli as analyze_profitability,
)
from src.tennis_betting_model.analysis.plot_tournament_leaderboard import (
    main_cli as plot_leaderboard,
)
from src.tennis_betting_model.pipeline.run_automation import main as run_automation


from src.tennis_betting_model.utils.logger import (
    setup_logging,
    log_info,
)
from src.tennis_betting_model.utils.config import load_config


def run_data_preparation_pipeline(args):
    """
    Prepares only the most raw, independent data sources.
    """
    log_info("--- Running Data Preparation Pipeline ---")
    config = load_config(args.config)
    log_info("\nStep 1: Consolidating player attributes...")
    consolidate_player_attributes(config)
    log_info("\nStep 2: Consolidating player rankings...")
    consolidate_rankings(config)
    log_info("\nStep 3: Building RAW odds from Betfair files...")
    build_raw_odds()
    log_info("\n--- Data Preparation Finished ---")


def run_player_map_pipeline(args):
    log_info("--- Running Player Mapping ---")
    config = load_config(args.config)
    create_mapping_file(config)


def run_build_pipeline(args):
    """
    Enriches data and builds all features.
    """
    log_info("--- Running Full Data Build ---")
    config = load_config(args.config)
    log_info("\nStep 1: Enriching data and creating match log...")
    build_match_log(config)
    log_info("\nStep 2: Calculating Elo ratings...")
    build_elo()
    log_info("\nStep 3: Building player features...")
    build_features(args)
    log_info("\nStep 4: Building clean data for realistic backtest...")
    build_backtest_data()
    log_info("\n--- Data Build Finished ---")


def main():
    parser = argparse.ArgumentParser(
        description="A comprehensive toolkit for tennis value betting analysis."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    p_prepare = subparsers.add_parser(
        "prepare-data", help="Prepare raw data sources (players, rankings, raw odds)."
    )
    p_prepare.set_defaults(func=run_data_preparation_pipeline)

    p_map = subparsers.add_parser(
        "create-player-map", help="Generate the player ID mapping file."
    )
    p_map.set_defaults(func=run_player_map_pipeline)

    p_build = subparsers.add_parser("build", help="Enrich data and build all features.")
    p_build.set_defaults(func=run_build_pipeline)

    p_model = subparsers.add_parser("model", help="Train and evaluate models.")
    p_model.set_defaults(func=train_model)

    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_backtest.add_argument(
        "mode", choices=["simulation", "realistic"], help="The backtesting mode to run."
    )
    p_backtest.set_defaults(func=run_backtest)

    p_analysis = subparsers.add_parser(
        "analysis", help="Run analysis on backtest results."
    )
    analysis_subparsers = p_analysis.add_subparsers(
        dest="analysis_command", required=True
    )

    p_summarize = analysis_subparsers.add_parser(
        "summarize-tournaments", help="Summarize results by tournament category."
    )
    p_summarize.add_argument(
        "--min-bets",
        type=int,
        default=100,
        help="Minimum bets to include a tournament category.",
    )
    p_summarize.add_argument(
        "--show-tournaments",
        action="store_true",
        help="Shows individual tournaments within each category.",
    )
    p_summarize.set_defaults(func=summarize_tournaments)

    p_profitability = analysis_subparsers.add_parser(
        "profitability", help="Analyze profitability of betting strategies."
    )
    p_profitability.set_defaults(func=analyze_profitability)

    p_plot = analysis_subparsers.add_parser(
        "plot-leaderboard", help="Plot tournament leaderboard by ROI."
    )
    p_plot.set_defaults(func=plot_leaderboard)

    p_pipeline = subparsers.add_parser(
        "pipeline", help="Run a single pipeline instance."
    )
    # --- REFACTOR: Add the --dry-run argument to the pipeline command ---
    p_pipeline.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without placing bets.",
    )
    # --- END REFACTOR ---
    p_pipeline.set_defaults(func=run_pipeline)

    p_automate = subparsers.add_parser(
        "automate", help="Run the pipeline on a schedule."
    )
    p_automate.set_defaults(func=run_automation)

    args = parser.parse_args()
    setup_logging()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
