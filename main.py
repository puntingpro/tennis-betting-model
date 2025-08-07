# main.py
import subprocess
import sys
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional
from argparse import Namespace

import typer

sys.path.append(str(Path(__file__).resolve().parent))

from src.tennis_betting_model.builders import (
    build_backtest_data,
    build_elo_ratings,
    build_enriched_odds,
    build_match_log,
    build_player_features,
    data_preparer,
    player_mapper,
)
from src.tennis_betting_model.modeling import train_eval_model
from src.tennis_betting_model.analysis import (
    analyze_profitability,
    list_tournaments,
    plot_tournament_leaderboard,
    run_backtest,
    summarize_value_bets_by_tournament,
)
from src.tennis_betting_model.pipeline import run_flumine
from src.tennis_betting_model.utils.logger import setup_logging, log_info, log_success
from src.tennis_betting_model.utils.config import load_config, Config

# --- Typer CLI Application Setup ---
app = typer.Typer(
    help="A comprehensive toolkit for tennis value betting analysis.",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

analysis_app = typer.Typer(
    help="Run analysis on backtest results.",
    add_completion=False,
)
app.add_typer(analysis_app, name="analysis")
# ---

# --- Global Options ---
ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to the configuration file.",
        default_factory=lambda: Path("config.yaml"),
        exists=True,
        readable=True,
    ),
]


# --- Main Commands ---
@app.command()
def stream(
    config_path: ConfigPath,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Run in dry-run mode without placing real money."
        ),
    ] = False,
):
    """Run the live, real-time trading bot using the Stream API."""
    # --- FIX: Add config_path to the Namespace object ---
    args = Namespace(dry_run=dry_run, config=str(config_path))
    run_flumine.main(args)


@app.command("prepare-data")
def run_data_preparation_pipeline(config_path: ConfigPath):
    """Prepare raw data sources (players, rankings, raw odds)."""
    log_info("--- Running Data Preparation Pipeline ---")
    config = Config(**load_config(str(config_path)))
    log_info("\nStep 1: Consolidating player attributes...")
    data_preparer.consolidate_player_attributes(config.data_paths)
    log_info("\nStep 2: Consolidating player rankings...")
    data_preparer.consolidate_rankings(config.data_paths)
    log_info("\nStep 3: Building RAW odds from Betfair summary files...")
    build_enriched_odds.main(config.data_paths)
    log_success("Data Preparation Finished")


@app.command("create-player-map")
def run_player_map_pipeline(config_path: ConfigPath):
    """Generate the player ID mapping file."""
    log_info("--- Running Player Mapping ---")
    config = Config(**load_config(str(config_path)))
    player_mapper.run_create_mapping_file(config.data_paths, config.mapping_params)
    log_success("Player Mapping Finished")


@app.command()
def build(config_path: ConfigPath):
    """Enrich data and build all features for training and backtesting."""
    log_info("--- Running Full Data Build Pipeline ---")
    config = Config(**load_config(str(config_path)))
    args = Namespace(config=str(config_path))

    log_info("\nStep 1: Creating historical match log from Betfair data...")
    build_match_log.main(config.data_paths)
    log_info("\nStep 2: Calculating surface-specific Elo ratings...")
    build_elo_ratings.main(config.data_paths, config.elo_config)
    log_info("\nStep 3: Building consolidated player features...")
    build_player_features.main(args)
    log_info("\nStep 4: Building clean market data for realistic backtesting...")
    build_backtest_data.main(config.data_paths)
    log_success("Data Build Finished")


@app.command()
def model(config_path: ConfigPath):
    """Train and evaluate the LightGBM model."""
    args = Namespace(config=str(config_path))
    train_eval_model.main_cli(args)


@app.command()
def backtest(
    mode: Annotated[
        str,
        typer.Argument(
            help="The backtesting mode to run: 'simulation' or 'realistic'."
        ),
    ],
    config_path: ConfigPath,
):
    """Run a historical backtest."""
    args = Namespace(mode=mode, config=str(config_path))
    run_backtest.main(args)


@app.command()
def dashboard(config_path: ConfigPath):
    """Launch the interactive performance dashboard."""
    log_info("Launching the Performance Dashboard...")
    dashboard_path = (
        Path(__file__).resolve().parent
        / "src/tennis_betting_model/dashboard/run_dashboard.py"
    )
    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--", "--config", str(config_path)],
        check=True,
    )


# --- Analysis Sub-Commands ---
@analysis_app.command("summarize-tournaments")
def summarize(config_path: ConfigPath):
    """Summarize results by tournament category."""
    args = Namespace(config=str(config_path))
    summarize_value_bets_by_tournament.main_cli(args)


@analysis_app.command("profitability")
def profitability(config_path: ConfigPath):
    """Analyze profitability of betting strategies from the config file."""
    args = Namespace(config=str(config_path))
    analyze_profitability.main_cli(args)


@analysis_app.command("plot-leaderboard")
def leaderboard(
    config_path: ConfigPath,
    show_plot: Annotated[
        bool, typer.Option("--show-plot", help="Display the plot interactively.")
    ] = False,
):
    """Plot tournament leaderboard by ROI."""
    args = Namespace(config=str(config_path), show_plot=show_plot)
    plot_tournament_leaderboard.main_cli(args)


@analysis_app.command("list-tournaments")
def list_tournaments_cmd(
    config_path: ConfigPath,
    year: Annotated[
        Optional[int], typer.Option("--year", help="Filter tournaments by year.")
    ] = None,
):
    """List all unique tournament names found in the data."""
    args = Namespace(config=str(config_path), year=year)
    list_tournaments.main_cli(args)


@analysis_app.command("review-mappings")
def review_mappings(config_path: ConfigPath):
    """Launch the interactive tool to review and correct player mappings."""
    log_info("Launching the Player Mapping Review Tool...")
    script_path = (
        Path(__file__).resolve().parent
        / "src/tennis_betting_model/analysis/review_player_mappings.py"
    )
    subprocess.run(
        ["streamlit", "run", str(script_path), "--", "--config", str(config_path)],
        check=True,
    )


@app.callback()
def main_callback():
    """
    Main callback to set up logging.
    """
    setup_logging()


if __name__ == "__main__":
    app()
