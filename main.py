# main.py
import sys
from pathlib import Path
from omegaconf import OmegaConf
import subprocess

# Add project root to path
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
from src.tennis_betting_model.utils.logger import (
    setup_logging,
    log_info,
    log_success,
    log_error,
)
from src.tennis_betting_model.utils.config import validate_config
from src.tennis_betting_model.utils.config_schema import Config


def main() -> None:
    """
    Main entry point for the tennis betting model with manual config loading.
    """
    # Manually parse command-line arguments
    args = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in sys.argv[1:]}

    # Manually load the single configuration file
    cfg = OmegaConf.load("conf/config.yaml")

    # Apply command-line overrides
    for key, value in args.items():
        OmegaConf.update(cfg, key, value)

    setup_logging()

    config_dict = validate_config(cfg)
    config = Config(**config_dict)

    command = cfg.get("command")
    if not command:
        log_error(
            "No command specified. Use 'command=<name>', e.g., 'python main.py command=prepare-data'"
        )
        return

    log_info(f"Running command: {command}")

    if command == "prepare-data":
        data_preparer.consolidate_player_attributes(config.data_paths)
        data_preparer.consolidate_rankings(config.data_paths)
        build_enriched_odds.main(config.data_paths)

    elif command == "create-player-map":
        player_mapper.run_create_mapping_file(config.data_paths, config.mapping_params)

    elif command == "build":
        build_match_log.main(config.data_paths)
        build_elo_ratings.main(config.data_paths, config.elo_config)
        build_backtest_data.main(config.data_paths)
        build_player_features.main(config)

    elif command == "model":
        train_eval_model.main_cli(config)

    elif command == "backtest":
        run_backtest.main(config, mode=cfg.get("mode", "realistic"))

    elif command == "stream":
        run_flumine.main(config, dry_run=cfg.get("dry_run", False))

    elif command == "dashboard":
        script_path = (
            Path(__file__).resolve().parent
            / "src/tennis_betting_model/dashboard/run_dashboard.py"
        )
        subprocess.run(["streamlit", "run", str(script_path)], check=True)

    elif command == "analysis/profitability":
        analyze_profitability.main_cli(config)

    elif command == "analysis/summarize-tournaments":
        summarize_value_bets_by_tournament.main_cli(config)

    elif command == "analysis/plot-leaderboard":
        plot_tournament_leaderboard.main_cli(
            config, show_plot=cfg.get("show_plot", False)
        )

    elif command == "analysis/list-tournaments":
        list_tournaments.main_cli(config, year=cfg.get("year"))

    else:
        log_error(f"Unknown command: {command}")
        return  # Add a return to prevent the success log on unknown command

    log_success(f"Command '{command}' finished successfully.")


if __name__ == "__main__":
    main()
