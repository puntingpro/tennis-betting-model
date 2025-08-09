# src/tennis_betting_model/builders/data_preparer.py

from pathlib import Path
import pandas as pd
import polars as pl
import glob

from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
)
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.utils.config_schema import DataPaths


def consolidate_player_attributes(paths: DataPaths):
    raw_data_dir = Path(paths.raw_data_dir)
    output_path = Path(paths.raw_players)
    atp_players_path = raw_data_dir / "tennis_atp" / "atp_players.csv"
    wta_players_path = raw_data_dir / "tennis_wta" / "wta_players.csv"
    if not atp_players_path.exists() or not wta_players_path.exists():
        log_error(
            f"Could not find player files at {atp_players_path} or {wta_players_path}."
        )
        return

    log_info("Loading ATP and WTA player attribute files...")
    player_cols = ["player_id", "first_name", "last_name", "hand", "dob", "country_ioc"]

    try:
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
    except pd.errors.EmptyDataError:
        log_error("One of the player attribute files is empty. Aborting.")
        return
    except ValueError as e:
        log_error(f"Error reading player attribute files: {e}. Check the file format.")
        return

    log_info("Combining files and removing duplicate player entries...")
    combined_df = pd.concat([df_atp, df_wta], ignore_index=True)

    combined_df["player_id"] = pd.to_numeric(combined_df["player_id"], errors="coerce")
    combined_df.dropna(subset=["player_id"], inplace=True)
    combined_df["player_id"] = combined_df["player_id"].astype(int)

    combined_df.drop_duplicates(subset=["player_id"], keep="first", inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    log_success(f"Consolidated {len(combined_df)} players into {output_path}")

    validate_data(combined_df, "raw_players", "Consolidated Player Attributes")


def consolidate_rankings(paths: DataPaths):
    raw_data_dir = Path(paths.raw_data_dir)
    output_path = Path(paths.consolidated_rankings)
    log_info("Finding all ATP and WTA ranking files...")
    all_files = glob.glob(str(raw_data_dir / "tennis_*" / "*_rankings_*.csv"))

    if not all_files:
        log_error("No ranking files found in the raw data directories.")
        return

    df_list = []
    for f in all_files:
        if Path(f).stat().st_size > 0:
            try:
                # Read all columns as string to avoid parsing errors
                # FIX: Removed the restrictive `dtypes` parameter to handle files with 4 or 5 columns
                df = pl.read_csv(f, has_header=False)
                # Filter out header rows
                df = df.filter(pl.col("column_1") != "ranking_date")

                if df.shape[1] == 4:
                    df.columns = ["ranking_date", "rank", "player", "points"]
                    df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("tours"))
                elif df.shape[1] == 5:
                    df.columns = ["ranking_date", "rank", "player", "points", "tours"]
                else:
                    log_error(
                        f"Skipping rankings file with incorrect number of columns: {f}"
                    )
                    continue
                df_list.append(df)
            except Exception as e:
                log_error(f"Could not process file: {f}. Error: {e}")

    if not df_list:
        log_error("No valid ranking files could be loaded.")
        return

    combined_df = pl.concat(df_list)

    combined_df = combined_df.with_columns(
        [
            pl.col("ranking_date")
            .str.to_datetime("%Y%m%d")
            .dt.replace_time_zone("UTC"),
            pl.col("rank").cast(pl.Int64, strict=False),
            pl.col("player").cast(pl.Int64, strict=False),
        ]
    ).drop_nulls(subset=["ranking_date", "rank", "player"])

    combined_df = combined_df.sort(by=["ranking_date", "rank"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.write_csv(output_path)
    log_success(f"Consolidated {len(combined_df)} ranking records into {output_path}")

    validate_data(
        combined_df.to_pandas(), "consolidated_rankings", "Consolidated Rankings"
    )
