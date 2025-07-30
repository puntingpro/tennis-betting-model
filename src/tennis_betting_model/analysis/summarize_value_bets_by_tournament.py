from pathlib import Path
import pandas as pd
import argparse
from tennis_betting_model.utils.file_utils import load_dataframes
from tennis_betting_model.utils.logger import log_success, setup_logging, log_error
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.common import get_tournament_category


def run_summarize_by_tournament(df: pd.DataFrame, min_bets: int = 1) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament category, calculating profit and ROI.
    """
    if df.empty:
        return pd.DataFrame()

    df["tourney_category"] = df["tourney_name"].apply(get_tournament_category)

    # --- REFACTOR: Use the pre-calculated 'pnl' column if it exists. ---
    if "pnl" not in df.columns:
        log_error(
            "Warning: 'pnl' column not found. Calculating profit without commission."
        )
        df["pnl"] = df.apply(
            lambda row: (row["odds"] - 1) if row["winner"] == 1 else -1, axis=1
        )

    df["stake"] = 1

    tournament_summary = (
        df.groupby("tourney_category")
        .agg(
            total_bets=("stake", "sum"),
            total_pnl=("pnl", "sum"),  # Use pnl column for aggregation
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

    df = load_dataframes(paths["backtest_results"])
    summary_df = run_summarize_by_tournament(df, args.min_bets)

    if not summary_df.empty:
        log_success("✅ Successfully summarized tournaments by category.")

        display_df = summary_df.copy()
        if not args.show_tournaments:
            display_df = display_df.drop(columns=["tournaments"])

        print(f"\n--- Tournament Category Performance (min_bets={args.min_bets}) ---")
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
    parser.add_argument(
        "--min-bets",
        type=int,
        default=100,
        help="The minimum number of total bets required to include a tournament category in the summary.",
    )
    parser.add_argument(
        "--show-tournaments",
        action="store_true",
        help="If set, shows the full list of individual tournaments within each category.",
    )
    args = parser.parse_args()
    main_cli(args)
