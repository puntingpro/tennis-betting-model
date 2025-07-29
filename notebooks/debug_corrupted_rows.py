# notebooks/debug_corrupted_rows.py
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import setup_logging, log_info


def debug_corruption():
    """
    Finds a corrupted row in the final feature file and traces it back
    to its source in the match log to identify the origin of the bug.
    """
    setup_logging()
    config = load_config("config.yaml")
    paths = config["data_paths"]
    features_path = Path(paths["consolidated_features"])
    match_log_path = Path(paths["betfair_match_log"])

    log_info(f"Loading features from: {features_path}")
    df_features = pd.read_csv(features_path)

    log_info(f"Loading match log from: {match_log_path}")
    df_log = pd.read_csv(match_log_path)

    # Find the first row where p1_id and p2_id are the same
    corrupted_row = df_features[df_features["p1_id"] == df_features["p2_id"]].iloc[0]

    if corrupted_row.empty:
        log_info("No corrupted rows found where p1_id == p2_id.")
        return

    corrupted_match_id = corrupted_row["match_id"]

    log_info(f"Found a corrupted row for match_id: {corrupted_match_id}")

    # Find the original entry in the match log
    original_row = df_log[df_log["match_id"] == corrupted_match_id]

    print("\n" + "=" * 50)
    log_info("CORRUPTED ROW from all_advanced_features.csv:")
    print(corrupted_row)
    print("=" * 50 + "\n")

    print("\n" + "=" * 50)
    log_info("ORIGINAL ROW from betfair_match_log.csv:")
    print(original_row.to_string())
    print("=" * 50 + "\n")


if __name__ == "__main__":
    debug_corruption()
