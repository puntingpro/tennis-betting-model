# main.py

import argparse
import sys
from pathlib import Path

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.scripts.builders.build_player_features import main as build_features_main
from src.scripts.modeling.train_eval_model import main_cli as train_model_main
from src.scripts.pipeline.run_pipeline import run_selective_value_pipeline
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
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Build Command ---
    p_build = subparsers.add_parser("build", help="Build advanced player features")
    p_build.set_defaults(func=lambda args: build_features_main())

    # --- Pipeline Command ---
    p_pipeline = subparsers.add_parser("pipeline", help="Run the live value-finding pipeline")
    p_pipeline.set_defaults(func=lambda args: run_selective_value_pipeline())

    # --- Model Command ---
    p_model = subparsers.add_parser("model", help="Train and evaluate models")
    p_model.add_argument("--input_glob", required=True, help="Glob pattern for feature CSVs.")
    p_model.add_argument("--output_model", required=True, help="Path to save the trained model file.")
    p_model.add_argument("--algorithm", choices=["xgb"], default="xgb", help="Algorithm to use.")
    p_model.add_argument("--test_size", type=float, default=0.2)
    p_model.set_defaults(func=train_model_main)
    
    # --- Backtest Command ---
    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    p_backtest.add_argument("--model_path", required=True)
    p_backtest.add_argument("--features_csv", required=True)
    p_backtest.add_argument("--output_csv", required=True)
    p_backtest.set_defaults(func=backtest_main)

    # --- Analysis Commands ---
    p_analysis = subparsers.add_parser("analysis", help="Run analysis on backtest results")
    analysis_subparsers = p_analysis.add_subparsers(dest="analysis_command", required=True)

    p_summarize_matches = analysis_subparsers.add_parser("summarize-matches", help="Summarize backtest results by match")
    p_summarize_matches.add_argument("--value_bets_glob", required=True)
    p_summarize_matches.add_argument("--output_dir", required=True)
    p_summarize_matches.set_defaults(func=summarize_matches_main)

    p_summarize_tourneys = analysis_subparsers.add_parser("summarize-tournaments", help="Summarize results by tournament")
    p_summarize_tourneys.add_argument("--input_glob", required=True)
    p_summarize_tourneys.add_argument("--output_csv", required=True)
    p_summarize_tourneys.set_defaults(func=summarize_tournaments_main)

    p_plot = analysis_subparsers.add_parser("plot-leaderboard", help="Plot tournament leaderboard")
    p_plot.add_argument("--input_csv", required=True)
    p_plot.add_argument("--output_png", default=None)
    p_plot.add_argument("--sort_by", default="roi")
    p_plot.add_argument("--top_n", type=int, default=25)
    p_plot.add_argument("--show", action="store_true")
    p_plot.set_defaults(func=plot_leaderboard_main)

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    args.func(args)

if __name__ == "__main__":
    main()