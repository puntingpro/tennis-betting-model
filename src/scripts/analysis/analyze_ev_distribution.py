import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.utils.file_utils import load_dataframes
from scripts.utils.logger import log_error, log_info, log_success


def run_analyze_ev_distribution(
    df: pd.DataFrame, ev_threshold: float, max_odds: float
) -> pd.DataFrame:
    """
    Analyzes and filters value bets based on EV and odds thresholds.
    """
    df_filtered = df[
        (df["expected_value"] > ev_threshold) & (df["odds"] < max_odds)
    ].copy()
    df_filtered["is_correct"] = (df_filtered["predicted_prob"] > 0.5) == df_filtered[
        "winner"
    ]
    return df_filtered


def plot_ev_distribution(df: pd.DataFrame, ev_threshold: float):
    """
    Generates a plot showing the distribution of Expected Value.
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


def main_cli(args):
    """
    Main CLI handler for analyzing EV distribution.
    Accepts args object from main.py.
    """
    try:
        df = load_dataframes(args.value_bets_glob)
        filtered_df = run_analyze_ev_distribution(
            df, args.ev_threshold, args.max_odds
        )

        if filtered_df.empty:
            log_info("No bets met the specified EV and odds criteria.")
            return

        roi = (
            (filtered_df["odds"] - 1).where(filtered_df["is_correct"], -1).sum()
            / len(filtered_df)
        ) * 100
        log_success(
            f"Found {len(filtered_df)} value bets with ROI: {roi:.2f}%"
        )

        if args.plot or args.save_plot:
            fig = plot_ev_distribution(df, args.ev_threshold)
            if args.save_plot:
                output_path = "data/plots/ev_distribution.png"
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