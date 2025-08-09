# src/tennis_betting_model/analysis/summarize_value_bets_by_tournament.py

from pathlib import Path
import pandas as pd
from src.tennis_betting_model.utils.file_utils import load_dataframes
from src.tennis_betting_model.utils.logger import log_success, setup_logging, log_error
from src.tennis_betting_model.utils.config_schema import Config
from src.tennis_betting_model.utils.common import get_tournament_category
from src.tennis_betting_model.utils.betting_math import calculate_pnl


def run_summarize_by_tournament(df: pd.DataFrame, min_bets: int = 1) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament category and surface,
    calculating profit, ROI, win rate, and average odds.
    """
    if df.empty or "surface" not in df.columns:
        log_error("DataFrame is empty or 'surface' column is missing.")
        return pd.DataFrame()

    df["tourney_category"] = df["tourney_name"].apply(get_tournament_category)
    df = calculate_pnl(df)
    df["stake"] = 1

    tournament_summary = (
        df.groupby(["tourney_category", "surface"])
        .agg(
            total_bets=("stake", "sum"),
            total_wins=("winner", "sum"),
            avg_odds=("odds", "mean"),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
    )

    tournament_summary["win_rate"] = (
        tournament_summary["total_wins"] / tournament_summary["total_bets"]
    ) * 100
    tournament_summary["roi"] = (
        tournament_summary["total_pnl"] / tournament_summary["total_bets"]
    ) * 100

    filtered_summary = tournament_summary[tournament_summary["total_bets"] >= min_bets]

    return filtered_summary.sort_values(by="roi", ascending=False)


def main_cli(config: Config) -> None:
    """
    Main CLI entrypoint for summarizing backtest results.
    """
    setup_logging()
    paths = config.data_paths
    analysis_params = config.analysis_params
    min_bets_threshold = analysis_params.min_bets_for_summary

    df = load_dataframes(paths.backtest_results)
    summary_df = run_summarize_by_tournament(df, min_bets_threshold)

    if not summary_df.empty:
        log_success("✅ Successfully summarized tournaments by category and surface.")

        display_cols = [
            "tourney_category",
            "surface",
            "total_bets",
            "win_rate",
            "avg_odds",
            "total_pnl",
            "roi",
        ]
        display_df = summary_df[display_cols].copy()
        display_df["win_rate"] = display_df["win_rate"].map("{:,.2f}%".format)
        display_df["avg_odds"] = display_df["avg_odds"].map("{:,.2f}".format)
        display_df["total_pnl"] = display_df["total_pnl"].map("{:,.2f}".format)
        display_df["roi"] = display_df["roi"].map("{:,.2f}%".format)

        print(f"\n--- Tournament Performance (min_bets={min_bets_threshold}) ---")
        print(display_df.to_string(index=False))

        output_path = Path(paths.tournament_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        log_success(f"✅ Saved summary data to {output_path}")
