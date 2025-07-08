# main.py

import argparse
import sys
from pathlib import Path

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.scripts.builders.build_player_features import main as build_features_main
from src.scripts.modeling.train_eval_model import main_cli as train_model_main
from src.scripts.pipeline.run_pipeline import main as pipeline_main
from src.scripts.analysis.backtest_strategy import main as backtest_main
from src.scripts.analysis.summarize_value_bets_by_match import main_cli as summarize_matches_main
from src.scripts.analysis.summarize_value_bets_by_tournament import main_cli as summarize_tournaments_main
from src.scripts.analysis.plot_tournament_leaderboard import main_cli as plot_leaderboard_main
from src.scripts.utils.logger import setup_logging

def main():
    """Main CLI entrypoint for the Tennis Value Betting Pipeline."""
    parser = argparse.ArgumentParser(
        description="A comprehensive toolkit for tennis value betting analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- CLI Commands ---
    p_build = subparsers.add_parser("build", help="Build advanced player features from raw data.")
    p_build.set_defaults(func=build_features_main)

    p_pipeline = subparsers.add_parser("pipeline", help="Run the live value-finding pipeline.")
    p_pipeline.set_defaults(func=pipeline_main)

    p_model = subparsers.add_parser("model", help="Train and evaluate models using features.")
    p_model.set_defaults(func=train_model_main)

    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_backtest.set_defaults(func=backtest_main)

    p_analysis = subparsers.add_parser("analysis", help="Run analysis on backtest results")
    analysis_subparsers = p_analysis.add_subparsers(dest="analysis_command", required=True)

    p_summarize_matches = analysis_subparsers.add_parser("summarize-matches", help="Summarize backtest results by match.")
    p_summarize_matches.set_defaults(func=summarize_matches_main)

    p_summarize_tourneys = analysis_subparsers.add_parser("summarize-tournaments", help="Summarize results by tournament.")
    p_summarize_tourneys.set_defaults(func=summarize_tournaments_main)

    p_plot = analysis_subparsers.add_parser("plot-leaderboard", help="Plot tournament leaderboard.")
    p_plot.set_defaults(func=plot_leaderboard_main)

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Call the function associated with the chosen command, passing the parsed args
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()