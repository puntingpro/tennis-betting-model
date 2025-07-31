from pathlib import Path
import pandas as pd
import argparse
from tennis_betting_model.utils.file_utils import load_dataframes
from tennis_betting_model.utils.logger import log_success, setup_logging
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.common import get_tournament_category
from tennis_betting_model.utils.betting_math import calculate_pnl


def run_summarize_by_tournament(df: pd.DataFrame, min_bets: int = 1) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament category, calculating profit and ROI.
    """
    if df.empty:
        return pd.DataFrame()

    df["tourney_category"] = df["tourney_name"].apply(get_tournament_category)

    df = calculate_pnl(df)

    df["stake"] = 1

    tournament_summary = (
        df.groupby("tourney_category")
        .agg(
            total_bets=("stake", "sum"),
            total_pnl=("pnl", "sum"),
            tournaments=("tourney_name", lambda x: sorted(x.unique().tolist())),
        )
        .reset_index()
    )
    tournament_summary["roi"] = (
        tournament_summary["total_pnl"] / tournament_summary["total_bets"]
    ) * 100

    filtered_summary = tournament_summary[tournament_summary["total_bets"] >= min_bets]

    return filtered_summary.sort_values(by="roi", ascending=False)


def main_cli(args: argparse.Namespace) -> None:
    """
    Main CLI entrypoint for summarizing backtest results by tournament category.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]
    # --- REFACTOR: Get min_bets from config file ---
    analysis_params = config.get("analysis_params", {})
    min_bets_threshold = analysis_params.get("min_bets_for_summary", 100)

    df = load_dataframes(paths["backtest_results"])
    summary_df = run_summarize_by_tournament(df, min_bets_threshold)

    if not summary_df.empty:
        log_success("✅ Successfully summarized tournaments by category.")

        display_df = summary_df.copy()
        if not args.show_tournaments:
            display_df = display_df.drop(columns=["tournaments"])

        print(
            f"\n--- Tournament Category Performance (min_bets={min_bets_threshold}) ---"
        )
        print(display_df.to_string(index=False))

        output_path = Path(paths["tournament_summary"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        log_success(f"✅ Saved summary data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize backtest results by tournament category to calculate profit and ROI."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    # --- REFACTOR: The --min-bets argument is removed as it's now controlled by config. ---
    parser.add_argument(
        "--show-tournaments",
        action="store_true",
        help="If set, shows the full list of individual tournaments within each category.",
    )
    args = parser.parse_args()
    main_cli(args)
