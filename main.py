# main.py

import argparse

# --- MODIFIED: Imported the Elo builder main function ---
from scripts.builders.consolidate_data import main as consolidate_data_main
from scripts.builders.consolidate_rankings import main as consolidate_rankings_main
from scripts.builders.build_elo_ratings import main as build_elo_ratings_main
from scripts.builders.build_player_features import main as build_features_main

# --- END MODIFICATION ---

from scripts.modeling.train_eval_model import main_cli as train_model_main
from scripts.pipeline.run_pipeline import main as pipeline_main
from scripts.analysis.backtest_strategy import main as backtest_main
from scripts.analysis.summarize_value_bets_by_tournament import (
    main_cli as summarize_tournaments_main,
)
from scripts.analysis.plot_tournament_leaderboard import (
    main_cli as plot_leaderboard_main,
)
from scripts.utils.logger import setup_logging
from scripts.pipeline.run_automation import main as automation_main


def consolidate_main(args):
    """Wrapper function to run all data consolidation scripts."""
    print("--- Running Data Consolidation ---")
    consolidate_data_main()
    consolidate_rankings_main()
    print("--- Data Consolidation Finished ---")


# --- ADDED: New wrapper function for the build step ---
def build_main(args):
    """Wrapper function to run all data building scripts in order."""
    print("--- Running Data Build ---")
    build_elo_ratings_main()
    build_features_main(args)
    print("--- Data Build Finished ---")


# --- END ADDITION ---


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

    p_consolidate = subparsers.add_parser("consolidate", help="Consolidate raw data.")
    p_consolidate.set_defaults(func=consolidate_main)

    # --- MODIFIED: Point the build command to the new wrapper function ---
    p_build = subparsers.add_parser(
        "build", help="Build advanced player features and Elo ratings."
    )
    p_build.set_defaults(func=build_main)
    # --- END MODIFICATION ---

    p_model = subparsers.add_parser("model", help="Train and evaluate models.")
    p_model.set_defaults(func=train_model_main)

    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_backtest.set_defaults(func=backtest_main)

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
    p_summarize.set_defaults(func=summarize_tournaments_main)
    p_plot = analysis_subparsers.add_parser(
        "plot-leaderboard", help="Plot tournament leaderboard."
    )
    p_plot.set_defaults(func=plot_leaderboard_main)

    p_pipeline = subparsers.add_parser(
        "pipeline", help="Run a single pipeline instance."
    )
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=pipeline_main)

    p_automate = subparsers.add_parser(
        "automate", help="Run the pipeline on a schedule."
    )
    p_automate.set_defaults(func=automation_main)

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
