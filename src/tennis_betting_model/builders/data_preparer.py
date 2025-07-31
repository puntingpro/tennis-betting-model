# src/tennis_betting_model/builders/data_preparer.py

from pathlib import Path
import pandas as pd
import glob

from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
)


def consolidate_player_attributes(config: dict):
    paths = config["data_paths"]
    raw_data_dir = Path(paths["raw_data_dir"])
    output_path = Path(paths["raw_players"])
    atp_players_path = raw_data_dir / "tennis_atp" / "atp_players.csv"
    wta_players_path = raw_data_dir / "tennis_wta" / "wta_players.csv"
    if not atp_players_path.exists() or not wta_players_path.exists():
        log_error(
            f"Could not find player files at {atp_players_path} or {wta_players_path}."
        )
        return

    log_info("Loading ATP and WTA player attribute files...")
    player_cols = ["player_id", "first_name", "last_name", "hand", "dob", "country_ioc"]

    df_atp = pd.read_csv(
        atp_players_path,
        header=None,
        encoding="latin-1",
        usecols=range(6),
        names=player_cols,
        dtype={"player_id": str, "dob": str},
    )
    df_wta = pd.read_csv(
        wta_players_path,
        header=None,
        encoding="latin-1",
        usecols=range(6),
        names=player_cols,
        dtype={"player_id": str, "dob": str},
    )

    log_info("Combining files and removing duplicate player entries...")
    combined_df = pd.concat([df_atp, df_wta], ignore_index=True)

    combined_df["player_id"] = pd.to_numeric(combined_df["player_id"], errors="coerce")
    combined_df.dropna(subset=["player_id"], inplace=True)
    combined_df["player_id"] = combined_df["player_id"].astype(int)

    combined_df.drop_duplicates(subset=["player_id"], keep="first", inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    log_success(f"Consolidated {len(combined_df)} players into {output_path}")


def consolidate_rankings(config: dict):
    paths = config["data_paths"]
    raw_data_dir = Path(paths["raw_data_dir"])
    output_path = Path(paths["consolidated_rankings"])
    log_info("Finding all ATP and WTA ranking files...")
    all_files = glob.glob(str(raw_data_dir / "tennis_*" / "*_rankings_*.csv"))

    if not all_files:
        log_error("No ranking files found in the raw data directories.")
        return

    df_list = []
    for f in all_files:
        if Path(f).stat().st_size > 0:
            df_list.append(pd.read_csv(f, header=None, dtype=str))

    combined_df = pd.concat(df_list, ignore_index=True)

    # --- FIX: Cast column lists to a pd.Index to resolve mypy error ---
    if combined_df.shape[1] == 4:
        combined_df.columns = pd.Index(["ranking_date", "rank", "player", "points"])
    elif combined_df.shape[1] == 5:
        combined_df.columns = pd.Index(
            ["ranking_date", "rank", "player", "points", "tours"]
        )

    combined_df["ranking_date"] = pd.to_datetime(
        combined_df["ranking_date"], errors="coerce", format="%Y%m%d", utc=True
    )
    combined_df["rank"] = pd.to_numeric(combined_df["rank"], errors="coerce")
    combined_df["player"] = pd.to_numeric(combined_df["player"], errors="coerce")
    combined_df.dropna(subset=["ranking_date", "rank", "player"], inplace=True)

    combined_df = combined_df.astype({"rank": int, "player": int})

    combined_df.sort_values(by=["ranking_date", "rank"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    log_success(f"Consolidated {len(combined_df)} ranking records into {output_path}")
