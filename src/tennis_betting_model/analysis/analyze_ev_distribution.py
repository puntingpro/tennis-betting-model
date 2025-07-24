# src/scripts/analysis/analyze_ev_distribution.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path

from scripts.utils.file_utils import load_dataframes
from scripts.utils.logger import log_error, log_info, log_success, setup_logging


def run_analyze_ev_distribution(
    df: pd.DataFrame, ev_threshold: float, max_odds: float
) -> pd.DataFrame:
    """
    Analyzes and filters value bets based on EV and odds thresholds.

    Args:
        df (pd.DataFrame): DataFrame of all identified value bets.
        ev_threshold (float): The minimum Expected Value to be included.
        max_odds (float): The maximum odds to be included.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only bets that meet the criteria.
    """
    df_filtered = df[
        (df["expected_value"] > ev_threshold) & (df["odds"] < max_odds)
    ].copy()
    df_filtered["is_correct"] = (df_filtered["predicted_prob"] > 0.5) == df_filtered[
        "winner"
    ]
    return df_filtered


def plot_ev_distribution(df: pd.DataFrame, ev_threshold: float) -> plt.Figure:
    """
    Generates and returns a plot showing the distribution of Expected Value.

    Args:
        df (pd.DataFrame): DataFrame containing expected_value data.
        ev_threshold (float): The EV threshold to display as a vertical line.

    Returns:
        plt.Figure: The matplotlib Figure object for the generated plot.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(df["expected_value"], bins=50, kde=True, ax=ax, color="skyblue")
    ax.axvline(
        ev_threshold,
        color="r",
        linestyle="--",
        label=f"EV Threshold ({ev_threshold:.2f})",
    )
    ax.set_title("Distribution of Expected Value (EV) for All Bets", fontsize=16)
    ax.set_xlabel("Expected Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


def main_cli() -> None:
    """
    Main CLI handler for analyzing the distribution of Expected Value in bet results.
    """
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the distribution of Expected Value from backtest results."
    )
    parser.add_argument(
        "value_bets_glob",
        help="Glob pattern for the input CSV files containing value bets.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional: Path to save the filtered DataFrame as a CSV.",
    )
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=0.1,
        help="The minimum EV to filter bets by.",
    )
    parser.add_argument(
        "--max-odds",
        type=float,
        default=10.0,
        help="The maximum odds to filter bets by.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="If set, display the EV distribution plot."
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="If set, save the EV distribution plot to 'data/plots/'.",
    )

    args = parser.parse_args()

    try:
        df = load_dataframes(args.value_bets_glob)
        filtered_df = run_analyze_ev_distribution(df, args.ev_threshold, args.max_odds)

        if filtered_df.empty:
            log_info("No bets met the specified EV and odds criteria.")
            return

        # Calculate ROI for the filtered bets
        roi = (
            (filtered_df["odds"] - 1).where(filtered_df["is_correct"], -1).sum()
            / len(filtered_df)
        ) * 100
        log_success(f"Found {len(filtered_df)} value bets with ROI: {roi:.2f}%")

        if args.plot or args.save_plot:
            fig = plot_ev_distribution(df, args.ev_threshold)
            if args.save_plot:
                output_path = Path("data/plots/ev_distribution.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=300)
                log_success(f"Saved plot to {output_path}")
            if args.plot:
                plt.show()

        if args.output_csv:
            filtered_df.to_csv(args.output_csv, index=False)
            log_success(f"Saved filtered value bets to {args.output_csv}")

    except FileNotFoundError as e:
        log_error(f"Error loading files: {e}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main_cli()
