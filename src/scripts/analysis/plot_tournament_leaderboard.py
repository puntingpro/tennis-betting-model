import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.utils.file_utils import load_dataframes
from scripts.utils.logger import log_error, log_info, log_success


def run_plot_leaderboard(df: pd.DataFrame, sort_by: str, top_n: int):
    """
    Generates and saves a leaderboard plot from tournament summary data.
    """
    df = df.sort_values(by=sort_by, ascending=False).head(top_n)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Changed df["tournament"] to df["label"]
    bars = plt.barh(df["label"], df[sort_by], edgecolor="black")

    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xlabel(sort_by.replace("_", " ").title())
    
    # Changed Y-axis label to "Label" for consistency
    plt.ylabel("Label")
    
    plt.title(f"Top {top_n} Tours by {sort_by.title()}", fontsize=16)
    plt.tight_layout()
    return fig


def main_cli(args):
    """
    Main CLI handler for plotting leaderboard.
    Accepts args object from main.py.
    """
    try:
        df = pd.read_csv(args.input_csv)
        fig = run_plot_leaderboard(df, sort_by=args.sort_by, top_n=args.top_n)
        
        if args.show:
            plt.show()

        if args.output_png:
            fig.savefig(args.output_png, dpi=300)
            log_success(f"Saved leaderboard plot to {args.output_png}")

    except FileNotFoundError:
        log_error(f"Input file not found: {args.input_csv}")
    except KeyError as e:
        log_error(f"Invalid column name for sorting: {e}. Check your --sort_by argument.")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")