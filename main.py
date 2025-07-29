import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.tennis_betting_model.builders.build_elo_ratings import main as build_elo
from src.tennis_betting_model.builders.build_player_features import (
    main as build_features,
)
from src.tennis_betting_model.builders.build_enriched_odds import (
    main as build_enriched_odds,
)
from scripts.create_match_log_from_betfair import create_match_log
from scripts.consolidate_players import consolidate_player_attributes
from scripts.consolidate_rankings import consolidate_rankings

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
from src.tennis_betting_model.utils.logger import setup_logging, log_info
from src.tennis_betting_model.pipeline.run_automation import main as run_automation


def run_data_preparation_pipeline(args):
    """Executes the full data preparation sequence."""
    log_info("--- Running Full Data Preparation Pipeline ---")
    log_info("\nStep 1: Consolidating player attributes...")
    consolidate_player_attributes()
    log_info("\nStep 2: Consolidating player rankings...")
    consolidate_rankings()
    log_info("\nStep 3: Building enriched odds from raw Betfair files...")
    build_enriched_odds()
    log_info("\nStep 4: Creating historical match log from enriched odds...")
    create_match_log()
    log_info("\n--- Data Preparation Finished ---")


def run_build_pipeline(args):
    """Executes the feature and Elo rating build sequence."""
    log_info("--- Running Data Build ---")
    build_elo()
    build_features(args)
    log_info("--- Data Build Finished ---")


def main():
    """Main CLI entrypoint for the Tennis Value Betting Pipeline."""
    parser = argparse.ArgumentParser(
        description="A comprehensive toolkit for tennis value betting analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed logging."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    p_prepare = subparsers.add_parser(
        "prepare-data", help="Run the entire data preparation pipeline."
    )
    p_prepare.set_defaults(func=run_data_preparation_pipeline)

    p_build = subparsers.add_parser(
        "build", help="Build advanced player features and Elo ratings."
    )
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
        "summarize-tournaments", help="Summarize results by tournament."
    )
    p_summarize.add_argument("--min-bets", type=int, default=100)
    p_summarize.add_argument("--show-tournaments", action="store_true")
    p_summarize.set_defaults(func=summarize_tournaments)
    p_profitability = analysis_subparsers.add_parser(
        "profitability", help="Analyze profitability of betting strategies."
    )
    p_profitability.set_defaults(func=analyze_profitability)
    p_plot = analysis_subparsers.add_parser(
        "plot-leaderboard", help="Plot tournament leaderboard."
    )
    p_plot.set_defaults(func=plot_leaderboard)

    p_pipeline = subparsers.add_parser(
        "pipeline", help="Run a single pipeline instance."
    )
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=run_pipeline)

    p_automate = subparsers.add_parser(
        "automate", help="Run the pipeline on a schedule."
    )
    p_automate.set_defaults(func=run_automation)

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
