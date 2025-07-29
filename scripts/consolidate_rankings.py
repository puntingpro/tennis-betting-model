from pathlib import Path
import pandas as pd
import glob
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    setup_logging,
    log_info,
    log_success,
    log_error,
)


def consolidate_rankings():
    """
    Consolidates all available ATP and WTA ranking files from the raw
    data directory into a single CSV file.
    """
    setup_logging()

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        raw_data_dir = Path(paths["raw_data_dir"])
        output_path = Path(paths["consolidated_rankings"])
    except (FileNotFoundError, KeyError) as e:
        log_error(f"Could not load configuration from config.yaml. Error: {e}")
        return

    log_info("Finding all ATP and WTA ranking files...")
    atp_ranking_files = glob.glob(
        str(raw_data_dir / "tennis_atp" / "atp_rankings_*.csv")
    )
    wta_ranking_files = glob.glob(
        str(raw_data_dir / "tennis_wta" / "wta_rankings_*.csv")
    )
    all_files = atp_ranking_files + wta_ranking_files

    if not all_files:
        log_error("No ranking files found in the raw data directories.")
        return

    log_info(f"Found {len(all_files)} ranking files to consolidate.")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, header=None)
            df_list.append(df)
        except pd.errors.EmptyDataError:
            log_error(f"Warning: Skipping empty ranking file: {f}")
            continue

    combined_df = pd.concat(df_list, ignore_index=True)

    # Assign standard column names
    # Handles files with/without the 'tours' column gracefully
    if combined_df.shape[1] == 4:
        combined_df.columns = ["ranking_date", "rank", "player", "points"]
    elif combined_df.shape[1] == 5:
        combined_df.columns = ["ranking_date", "rank", "player", "points", "tours"]

    combined_df["ranking_date"] = pd.to_datetime(
        combined_df["ranking_date"], format="%Y%m%d"
    )
    combined_df.sort_values(by=["ranking_date", "rank"], inplace=True)

    # Ensure output directory exists and save the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    log_success(
        f"Successfully consolidated {len(combined_df)} ranking records into {output_path}"
    )


if __name__ == "__main__":
    consolidate_rankings()
