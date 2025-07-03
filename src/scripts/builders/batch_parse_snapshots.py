from pathlib import Path
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.file_utils import write_csv
from scripts.utils.logger import log_error, log_info, log_success, log_warning
from scripts.utils.snapshot_parser import SnapshotParser


def find_and_parse_files_for_tournament(
    tournament_config: dict, raw_data_dir: Path, parser: SnapshotParser
) -> pd.DataFrame:
    """
    Finds all raw data files for a given tournament within its date range,
    parses them, and returns a single combined DataFrame.
    """
    label = tournament_config["label"]
    try:
        start_date = pd.to_datetime(tournament_config["start_date"])
        end_date = pd.to_datetime(tournament_config["end_date"])
    except KeyError:
        log_warning(
            f"Skipping '{label}': missing 'start_date' or 'end_date' in config."
        )
        return pd.DataFrame()

    log_info(
        f"Scanning for '{label}' files from {start_date.date()} to {end_date.date()}..."
    )

    all_records = []
    # Generate a list of dates to check
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    for date in date_range:
        # Construct path based on the structure: {Year}/{Month Abbreviation}/{Day}
        date_path = raw_data_dir / str(date.year) / date.strftime("%b") / str(date.day)

        if not date_path.exists():
            continue

        # Glob for all market files within that day's directory
        market_files = list(date_path.glob("**/*"))
        # Filter out directories, keeping only files
        market_files = [f for f in market_files if f.is_file()]

        for file_path in market_files:
            try:
                # The parse_file method handles bz2 and plain text
                records = parser.parse_file(str(file_path))
                if records:
                    all_records.extend(records)
            except Exception as e:
                log_warning(f"Could not parse file {file_path}: {e}")

    if not all_records:
        log_warning(f"No records found for tournament '{label}'.")
        return pd.DataFrame()

    log_success(f"Found and parsed {len(all_records)} total records for '{label}'.")
    return pd.DataFrame(all_records)


def main_cli(args):
    """
    Batch parse raw Betfair snapshot files based on a tournament config.
    The `args` object is now passed from the main CLI entrypoint.
    """
    try:
        app_cfg = load_config(args.config)
        tournaments = app_cfg.get("tournaments", [])
        if not tournaments:
            raise ValueError("No 'tournaments' found in the config file.")
    except Exception as e:
        log_error(f"Failed to load or validate config file {args.config}: {e}")
        return

    # Use the 'full' mode parser, which is what the pipeline expects
    snapshot_parser = SnapshotParser(mode="full")
    raw_data_path = Path(args.raw_data_dir)

    for tournament in tournaments:
        label = tournament.get("label", "Unknown")
        output_csv_path = Path(tournament.get("snapshots_csv", ""))

        if not output_csv_path:
            log_warning(f"Skipping '{label}': missing 'snapshots_csv' key in config.")
            continue

        if output_csv_path.exists() and not args.overwrite:
            log_info(
                f"Skipping '{label}': output file {output_csv_path} already exists."
            )
            continue

        # Ensure the output directory (e.g., 'parsed/') exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = find_and_parse_files_for_tournament(
            tournament, raw_data_path, snapshot_parser
        )

        if not df.empty:
            try:
                write_csv(df, str(output_csv_path), overwrite=args.overwrite)
                log_success(
                    f"Successfully saved combined data for '{label}' to {output_csv_path}"
                )
            except Exception as e:
                log_error(f"Failed to write CSV for '{label}': {e}")