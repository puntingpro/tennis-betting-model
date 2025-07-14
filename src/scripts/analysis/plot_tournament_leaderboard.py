# src/scripts/analysis/plot_tournament_leaderboard.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from src.scripts.utils.logger import log_error, log_success, setup_logging
from src.scripts.utils.config import load_config

def run_plot_leaderboard(df: pd.DataFrame, sort_by: str, top_n: int):
    df = df.sort_values(by=sort_by, ascending=False).head(top_n)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    bars = plt.barh(df["tourney_name"], df[sort_by], edgecolor="black")

    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xlabel(sort_by.replace("_", " ").title())
    plt.ylabel("Tournament")
    plt.title(f"Top {top_n} Tournaments by {sort_by.title()}", fontsize=16)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    return fig

def main_cli(args):
    """
    Main function for plotting the leaderboard, driven by the config file.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']

    try:
        input_path = Path(paths['tournament_summary'])
        df = pd.read_csv(input_path)
        
        # Default values from the old parser can be hardcoded or moved to config
        fig = run_plot_leaderboard(df, sort_by="roi", top_n=25)
        
        # --- MODIFIED: Use pathlib for robust path creation ---
        plot_dir = Path(paths.get('plot_dir', 'data/plots/'))
        output_path = plot_dir / 'tournament_leaderboard.png'
        
        # Create the parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=300)
        log_success(f"Saved leaderboard plot to {output_path}")
        
        # Show the plot in a window
        plt.show()

    except FileNotFoundError:
        log_error(f"Input file not found: {paths['tournament_summary']}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main_cli(args)