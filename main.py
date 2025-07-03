import argparse
import sys

from scripts.analysis.analyze_ev_distribution import main_cli as analyze_ev_main
from scripts.analysis.plot_tournament_leaderboard import (
    main_cli as plot_leaderboard_main,
)
from scripts.analysis.summarize_value_bets_by_match import (
    main_cli as summarize_matches_main,
)
from scripts.analysis.summarize_value_bets_by_tournament import (
    main_cli as summarize_tournaments_main,
)
from scripts.builders.batch_parse_snapshots import main_cli as batch_parse_main
from scripts.builders.core import main as build_main
from scripts.modeling.train_ev_filter_model import main_cli as train_filter_main
from scripts.modeling.train_eval_model import main_cli as train_eval_main
from scripts.pipeline.run_full_pipeline import main as run_pipeline_main
from scripts.utils.constants import DEFAULT_EV_THRESHOLD, DEFAULT_MAX_ODDS
from scripts.utils.logger import setup_logging

def main():
    """Main CLI entrypoint for the Tennis Value Betting Pipeline."""
    parser = argparse.ArgumentParser(
        description="A comprehensive toolkit for tennis value betting analysis and simulation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Global args for logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose error logging.")
    parser.add_argument("--json_logs", action="store_true", help="Output logs in JSON format.")

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Batch Parse Command ---
    p_batch_parse = subparsers.add_parser(
        "batch-parse", help="Batch parse raw snapshots based on a tournament config"
    )
    p_batch_parse.add_argument(
        "--config",
        required=True,
        help="Path to the tournaments config YAML file.",
    )
    p_batch_parse.add_argument(
        "--raw_data_dir",
        default="data/BASIC",
        help="Root directory of the raw Betfair data.",
    )
    p_batch_parse.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parsed CSV files.",
    )
    p_batch_parse.set_defaults(func=batch_parse_main)

    # --- Build Command (from core) ---
    p_build = subparsers.add_parser(
        "build", help="Build matches from a single parsed snapshot CSV"
    )
    p_build.add_argument(
        "--input_path", required=True, help="Path to the parsed snapshot CSV."
    )
    p_build.add_argument(
        "--output_path", required=True, help="Path to save the output matches CSV."
    )
    p_build.add_argument(
        "--dry_run", action="store_true", help="Log actions without writing files."
    )
    p_build.set_defaults(func=build_main)


    # --- Pipeline Command ---
    p_pipeline = subparsers.add_parser(
        "pipeline", help="Run the data processing pipeline"
    )
    p_pipeline.add_argument(
        "--config",
        default="configs/pipeline_run.yaml",
        help="Path to the main pipeline or batch config YAML file.",
    )
    p_pipeline.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the specified pipeline stages.",
    )
    p_pipeline.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode for all tournaments in the config file.",
    )
    p_pipeline.add_argument(
        "--dry_run", action="store_true", help="Log actions without writing files."
    )
    p_pipeline.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files for each stage.",
    )
    p_pipeline.add_argument(
        "--working_dir",
        default="data/processed",
        help="Directory for intermediate and final pipeline outputs.",
    )
    p_pipeline.set_defaults(func=run_pipeline_main)

    # --- Modeling Commands ---
    p_model = subparsers.add_parser("model", help="Train and evaluate models")
    model_subparsers = p_model.add_subparsers(dest="model_command", required=True)

    p_train_eval = model_subparsers.add_parser(
        "train-eval", help="Train and evaluate a general classification model"
    )
    p_train_eval.add_argument(
        "--input_glob",
        required=True,
        help="Glob pattern for value bet CSVs for training.",
    )
    p_train_eval.add_argument(
        "--output_model", required=True, help="Path to save the trained model file."
    )
    p_train_eval.add_argument(
        "--algorithm",
        choices=["rf", "logreg", "xgb"],
        default="rf",
        help="Algorithm to use for training.",
    )
    p_train_eval.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Fraction of data to use for the test set.",
    )
    p_train_eval.set_defaults(func=train_eval_main)

    p_train_filter = model_subparsers.add_parser(
        "train-filter", help="Train the simpler EV filter model"
    )
    p_train_filter.add_argument(
        "--input_glob",
        required=True,
        help="Glob pattern for value bet CSVs for training.",
    )
    p_train_filter.add_argument(
        "--output_model", required=True, help="Path to save the trained model file."
    )
    p_train_filter.add_argument("--min_ev", type=float, default=DEFAULT_EV_THRESHOLD)
    p_train_filter.set_defaults(func=train_filter_main)

    # --- Analysis Commands ---
    p_analysis = subparsers.add_parser(
        "analysis", help="Run analysis and generate plots"
    )
    analysis_subparsers = p_analysis.add_subparsers(
        dest="analysis_command", required=True
    )

    p_analyze_ev = analysis_subparsers.add_parser(
        "analyze-ev", help="Analyze and plot EV distribution"
    )
    p_analyze_ev.add_argument(
        "--value_bets_glob", required=True, help="Glob pattern for value bet CSVs."
    )
    p_analyze_ev.add_argument(
        "--output_csv", default=None, help="Path to save filtered bets."
    )
    p_analyze_ev.add_argument(
        "--ev_threshold", type=float, default=DEFAULT_EV_THRESHOLD
    )
    p_analyze_ev.add_argument("--max_odds", type=float, default=DEFAULT_MAX_ODDS)
    p_analyze_ev.add_argument(
        "--plot", action="store_true", help="Display the EV distribution plot."
    )
    p_analyze_ev.add_argument(
        "--save_plot",
        action="store_true",
        help="Save the EV distribution plot to a file.",
    )
    p_analyze_ev.set_defaults(func=analyze_ev_main)

    p_summarize_matches = analysis_subparsers.add_parser(
        "summarize-matches", help="Summarize value bets by match"
    )
    p_summarize_matches.add_argument(
        "--value_bets_glob", required=True, help="Glob pattern for value bet CSVs."
    )
    p_summarize_matches.add_argument(
        "--output_dir", default="data/analysis/match_summaries", help="Directory to save the match-level summaries."
    )
    p_summarize_matches.set_defaults(func=summarize_matches_main)

    p_summarize_tourneys = analysis_subparsers.add_parser(
        "summarize-tournaments", help="Summarize results by tournament"
    )
    p_summarize_tourneys.add_argument(
        "--input_glob", required=True, help="Glob pattern for match-level summary CSVs."
    )
    p_summarize_tourneys.add_argument(
        "--output_csv", required=True, help="Path to save the tournament-level summary."
    )
    p_summarize_tourneys.set_defaults(func=summarize_tournaments_main)

    p_plot = analysis_subparsers.add_parser(
        "plot-leaderboard", help="Plot tournament leaderboard"
    )
    p_plot.add_argument(
        "--input_csv", required=True, help="Path to the tournament summary CSV."
    )
    p_plot.add_argument(
        "--output_png", default=None, help="Path to save the output plot."
    )
    p_plot.add_argument("--sort_by", default="roi")
    p_plot.add_argument("--top_n", type=int, default=20)
    p_plot.add_argument(
        "--show", action="store_true", help="Display the plot interactively."
    )
    p_plot.set_defaults(func=plot_leaderboard_main)

    args = parser.parse_args()
    
    # Setup logging globally based on top-level args
    setup_logging(level="DEBUG" if args.verbose else "INFO", json_logs=args.json_logs)

    # All functions now accept the 'args' object.
    if args.command == "pipeline":
        # The pipeline function expects specific keyword arguments.
        run_pipeline_main(
            config=args.config,
            only=args.only,
            batch=args.batch,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            verbose=args.verbose,
            json_logs=args.json_logs,
            working_dir=args.working_dir,
        )
    else:
        # All other functions are designed to receive the raw `args' object.
        args.func(args)


if __name__ == "__main__":
    main()