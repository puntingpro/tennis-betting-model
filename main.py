import argparse
import sys
from pathlib import Path
import subprocess

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
from src.tennis_betting_model.analysis.list_tournaments import (
    main_cli as list_tournaments,
)
from src.tennis_betting_model.pipeline.run_flumine import main as run_stream


from src.tennis_betting_model.utils.logger import (
    setup_logging,
    log_info,
    log_success,
)
from src.tennis_betting_model.utils.config import load_config


def run_data_preparation_pipeline(args):
    """Orchestrates the initial data consolidation steps."""
    log_info("--- Running Data Preparation Pipeline ---")
    config = load_config(args.config)
    log_info("\nStep 1: Consolidating player attributes...")
    consolidate_player_attributes(config)
    log_info("\nStep 2: Consolidating player rankings...")
    consolidate_rankings(config)
    log_info("\nStep 3: Building RAW odds from Betfair summary files...")
    build_raw_odds()
    log_success("\n--- Data Preparation Finished ---")


def run_player_map_pipeline(args):
    """Runs the standalone player mapping generation."""
    log_info("--- Running Player Mapping ---")
    config = load_config(args.config)
    create_mapping_file(config)
    log_success("\n--- Player Mapping Finished ---")


def run_build_pipeline(args):
    """Orchestrates all data enrichment and feature engineering steps."""
    log_info("--- Running Full Data Build Pipeline ---")
    config = load_config(args.config)
    log_info("\nStep 1: Creating historical match log from Betfair data...")
    build_match_log(config)
    log_info("\nStep 2: Calculating surface-specific Elo ratings...")
    build_elo()
    log_info("\nStep 3: Building consolidated player features...")
    build_features(args)
    log_info("\nStep 4: Building clean market data for realistic backtesting...")
    build_backtest_data()
    log_success("\n--- Data Build Finished ---")


def run_dashboard_command(args):
    """Launches the Streamlit performance dashboard."""
    log_info("Launching the Performance Dashboard...")
    dashboard_path = (
        Path(__file__).resolve().parent
        / "src"
        / "tennis_betting_model"
        / "dashboard"
        / "run_dashboard.py"
    )
    subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)


def run_review_mappings_command(args):
    """Launches the Streamlit Player Mapping Review tool."""
    log_info("Launching the Player Mapping Review Tool...")
    script_path = (
        Path(__file__).resolve().parent
        / "src"
        / "tennis_betting_model"
        / "analysis"
        / "review_player_mappings.py"
    )
    subprocess.run(["streamlit", "run", str(script_path)], check=True)


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

    # --- Command to Function Mapping ---
    command_functions = {
        "stream": run_stream,
        "prepare-data": run_data_preparation_pipeline,
        "create-player-map": run_player_map_pipeline,
        "build": run_build_pipeline,
        "model": train_model,
        "backtest": run_backtest,
        "summarize-tournaments": summarize_tournaments,
        "profitability": analyze_profitability,
        "plot-leaderboard": plot_leaderboard,
        "list-tournaments": list_tournaments,
        "review-mappings": run_review_mappings_command,
        "dashboard": run_dashboard_command,
    }

    # --- Define Parsers ---
    p_stream = subparsers.add_parser(
        "stream", help="Run the live, real-time trading bot using the Stream API."
    )
    p_stream.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode to see potential bets without placing real money.",
    )

    subparsers.add_parser(
        "prepare-data", help="Prepare raw data sources (players, rankings, raw odds)."
    )
    subparsers.add_parser(
        "create-player-map", help="Generate the player ID mapping file."
    )
    subparsers.add_parser("build", help="Enrich data and build all features.")
    subparsers.add_parser("model", help="Train and evaluate the LightGBM model.")
    p_backtest = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_backtest.add_argument(
        "mode", choices=["simulation", "realistic"], help="The backtesting mode to run."
    )

    p_analysis = subparsers.add_parser(
        "analysis", help="Run analysis on backtest results."
    )
    analysis_subparsers = p_analysis.add_subparsers(
        dest="analysis_command", required=True
    )
    analysis_subparsers.add_parser(
        "summarize-tournaments", help="Summarize results by tournament category."
    )
    analysis_subparsers.add_parser(
        "profitability", help="Analyze profitability of betting strategies."
    )
    p_plot = analysis_subparsers.add_parser(
        "plot-leaderboard", help="Plot tournament leaderboard by ROI."
    )
    p_plot.add_argument(
        "--show-plot",
        action="store_true",
        help="If set, displays the leaderboard plot interactively.",
    )
    p_list = analysis_subparsers.add_parser(
        "list-tournaments",
        help="List all unique tournament names found in the data.",
    )
    p_list.add_argument(
        "--year",
        type=int,
        help="Optional: Filter tournaments by a specific year.",
    )
    analysis_subparsers.add_parser(
        "review-mappings",
        help="Launch the interactive tool to review and correct player mappings.",
    )
    subparsers.add_parser(
        "dashboard", help="Launch the interactive performance dashboard."
    )

    args = parser.parse_args()
    setup_logging()

    # --- Execute Command ---
    command = args.command
    if args.command == "analysis":
        command = args.analysis_command

    if command in command_functions:
        command_functions[command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
