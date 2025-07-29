# notebooks/debug_surface_logic.py
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tennis_betting_model.builders.build_player_features import get_surface
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import setup_logging, log_info


def debug_surfaces():
    """
    Loads the match log to analyze the effectiveness of the get_surface function.
    """
    setup_logging()
    config = load_config("config.yaml")
    paths = config["data_paths"]
    match_log_path = Path(paths["betfair_match_log"])

    log_info(f"Loading match log from: {match_log_path}")
    df = pd.read_csv(match_log_path)

    log_info("Applying current get_surface logic to all tournament names...")
    df["derived_surface"] = df["tourney_name"].apply(get_surface)

    print("\n" + "=" * 50)
    log_info("Current Surface Distribution:")
    print(df["derived_surface"].value_counts())
    print("=" * 50 + "\n")

    log_info("Investigating misclassified surfaces.")
    log_info("The following tournament names were defaulted to 'Hard':")

    # Get a unique list of tournament names that were defaulted to Hard
    misclassified_names = df[df["derived_surface"] == "Hard"]["tourney_name"].unique()

    # Print a sample of 50 to review
    for name in sorted(misclassified_names)[:50]:
        print(f"- {name}")


if __name__ == "__main__":
    debug_surfaces()
