# main.py

import argparse
import sys
from pathlib import Path

# --- MODIFIED: Corrected import paths for consolidation scripts ---
from src.scripts.builders.consolidate_data import main as consolidate_data_main
from src.scripts.builders.consolidate_rankings import main as consolidate_rankings_main
from src.scripts.builders.build_player_features import main as build_features_main
from src.scripts.modeling.train_eval_model import main_cli as train_model_main
from src.scripts.pipeline.run_pipeline import main as pipeline_main
from src.scripts.analysis.backtest_strategy import main as backtest_main
from src.scripts.analysis.summarize_value_bets_by_tournament import main_cli as summarize_tournaments_main
from src.scripts.analysis.plot_tournament_leaderboard import main_cli as plot_leaderboard_main
from src.scripts.utils.logger import setup_logging
from src.scripts.dashboard.run_dashboard import main as dashboard_main
from src.scripts.pipeline.run_automation import main as automation_main

def consolidate_main(args):
    """Wrapper function to run all data consolidation scripts."""
    print("--- Running Data Consolidation ---")
    consolidate_data_main()
    consolidate_rankings_main()
    print("--- Data Consolidation Finished ---")

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
    p_consolidate = subparsers.add_parser("consolidate", help="Consolidate all raw data files.")
    p_consolidate.set_defaults(func=consolidate_main)

    p_build = subparsers.add_parser("build", help="Build advanced player features from raw data.")
    p_build.set_defaults(func=build_features_main)
    
    p_model = subparsers.add_parser("model", help="Train and evaluate models using features.")
    p_model.set_defaults(func=train_model_main)
    
    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_backtest.set_defaults(func=backtest_main)

    p_analysis = subparsers.add_parser("analysis", help="Run analysis on backtest results")
    analysis_subparsers = p_analysis.add_subparsers(dest="analysis_command", required=True)
    p_summarize_tourneys = analysis_subparsers.add_parser("summarize-tournaments", help="Summarize results by tournament.")
    p_summarize_tourneys.set_defaults(func=summarize_tournaments_main)
    p_plot = analysis_subparsers.add_parser("plot-leaderboard", help="Plot tournament leaderboard.")
    p_plot.set_defaults(func=plot_leaderboard_main)
    
    p_dashboard = subparsers.add_parser("dashboard", help="Launch the project dashboard.")
    p_dashboard.set_defaults(func=lambda args: dashboard_main())
    
    p_automate = subparsers.add_parser("automate", help="Run the pipeline on an automated schedule.")
    p_automate.set_defaults(func=automation_main)


    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()