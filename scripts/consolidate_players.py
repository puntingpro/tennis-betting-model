from pathlib import Path
import pandas as pd
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


def consolidate_player_attributes():
    """
    Consolidates separate ATP and WTA player attribute files from the raw
    data directory into a single, unified player file.
    """
    setup_logging()

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        raw_data_dir = Path(paths["raw_data_dir"])
        output_path = Path(paths["raw_players"])
    except (FileNotFoundError, KeyError) as e:
        log_error(f"Could not load configuration from config.yaml. Error: {e}")
        return

    atp_players_path = raw_data_dir / "tennis_atp" / "atp_players.csv"
    wta_players_path = raw_data_dir / "tennis_wta" / "wta_players.csv"

    if not atp_players_path.exists() or not wta_players_path.exists():
        log_error(
            f"Could not find player files at {atp_players_path} or {wta_players_path}."
        )
        return

    log_info("Loading ATP and WTA player attribute files...")
    df_atp = pd.read_csv(atp_players_path, header=None, encoding="latin-1")
    df_wta = pd.read_csv(wta_players_path, header=None, encoding="latin-1")

    # Define standard column headers
    player_cols = ["player_id", "first_name", "last_name", "hand", "dob", "country_ioc"]
    df_atp.columns = player_cols
    df_wta.columns = player_cols

    log_info("Combining files and removing duplicate player entries...")
    combined_df = pd.concat([df_atp, df_wta], ignore_index=True)

    # Keep the first entry for any player_id that might appear in both
    combined_df.drop_duplicates(subset=["player_id"], keep="first", inplace=True)

    # Ensure output directory exists and save the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    log_success(
        f"Successfully consolidated {len(combined_df)} players into {output_path}"
    )


if __name__ == "__main__":
    consolidate_player_attributes()
