# src/tennis_betting_model/builders/build_enriched_odds.py

import tarfile
import orjson as json
import pandas as pd
import bz2
import os
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    log_warning,
    setup_logging,
)


def process_tar_file(tar_path, player_map):
    """
    Reads a raw Betfair .tar file, enriches the data line-by-line using the
    provided player map, and returns a list of processed market data.
    """
    market_data = []
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member_info in tar.getmembers():
                if not member_info.isfile():
                    continue
                file_obj = tar.extractfile(member_info)
                if file_obj:
                    with bz2.open(file_obj, "rt", encoding="utf-8") as bz2f:
                        for line in bz2f:
                            try:
                                change = json.loads(line)
                                if "mc" not in change:
                                    continue
                                for market_change in change["mc"]:
                                    market_def = market_change.get(
                                        "marketDefinition", {}
                                    )
                                    if market_def.get("marketType") != "MATCH_ODDS":
                                        continue
                                    tourney_name = market_def.get("eventName", "")
                                    if "/" in tourney_name:
                                        continue
                                    for runner_change in market_change.get("rc", []):
                                        atb = runner_change.get("atb", [])
                                        if atb:
                                            runner_id = runner_change.get("id")
                                            historical_id = player_map.get(runner_id)
                                            runner_info = next(
                                                (
                                                    r
                                                    for r in market_def.get(
                                                        "runners", []
                                                    )
                                                    if r.get("id") == runner_id
                                                ),
                                                None,
                                            )
                                            if runner_info:
                                                market_data.append(
                                                    {
                                                        "pt": change.get("pt"),
                                                        "market_id": market_change.get(
                                                            "id"
                                                        ),
                                                        "tourney_name": tourney_name,
                                                        "runner_id": runner_id,
                                                        "historical_id": historical_id,
                                                        "runner_name": runner_info.get(
                                                            "name"
                                                        ),
                                                        "best_back_price": atb[0][0],
                                                    }
                                                )
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
    except tarfile.ReadError:
        log_error(f"Could not read {tar_path}, it may be corrupted. Skipping.")

    return market_data


def main():
    """
    Main function to run the incremental data extraction and enrichment pipeline.
    """
    setup_logging()

    config = load_config("config.yaml")
    paths = config["data_paths"]
    raw_data_path = paths["raw_data_dir"]
    output_path = Path(paths["betfair_odds"])
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    mapping_path = Path(paths["player_map"])

    # State file to track processed TARs
    state_file = processed_dir / "processed_files.log"

    if not mapping_path.exists():
        log_error(
            f"Player mapping file not found at {mapping_path}. Please create it first."
        )
        return

    log_info(f"Loading player mapping from {mapping_path}...")
    df_map = pd.read_csv(mapping_path)
    df_map.dropna(subset=["betfair_id", "historical_id"], inplace=True)
    df_map["betfair_id"] = df_map["betfair_id"].astype(int)

    player_map = pd.Series(
        df_map.historical_id.values, index=df_map.betfair_id
    ).to_dict()
    log_info("Player map loaded successfully.")

    # --- Incremental Logic Start ---
    all_tar_files = set(glob.glob(os.path.join(raw_data_path, "*_ProTennis.tar")))

    processed_files = set()
    if state_file.exists():
        with state_file.open("r") as f:
            processed_files = {line.strip() for line in f}

    files_to_process = sorted(list(all_tar_files - processed_files))

    if not files_to_process:
        log_success("✅ No new TAR files to process. Data is already up-to-date.")
        return

    log_info(f"Found {len(files_to_process)} new TAR files to process.")
    # --- Incremental Logic End ---

    all_market_data = []
    worker_function = partial(process_tar_file, player_map=player_map)

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(worker_function, files_to_process),
                total=len(files_to_process),
                desc="Processing New TAR files",
            )
        )
        for result in results:
            all_market_data.extend(result)

    if not all_market_data:
        log_warning("No new market data was extracted from the new files.")
        return

    log_info("\nConsolidating new data...")
    new_df = pd.DataFrame(all_market_data)
    new_df.dropna(subset=["best_back_price"], inplace=True)
    new_df["pt"] = pd.to_datetime(new_df["pt"], unit="ms")

    # --- Combine with existing data if it exists ---
    if output_path.exists() and not processed_files:
        log_info(
            "State file not found, but output file exists. Forcing a full rebuild to ensure data integrity."
        )
    elif output_path.exists():
        log_info(f"Loading existing data from {output_path} to append new results...")
        existing_df = pd.read_csv(output_path, parse_dates=["pt"])
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.sort_values(by="pt", inplace=True)
    final_df.drop_duplicates(
        subset=["market_id", "runner_id", "pt"], keep="last", inplace=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    # --- Update state file with newly processed files ---
    with state_file.open("a") as f:
        for file_path in files_to_process:
            f.write(f"{file_path}\n")

    log_success(
        f"Pipeline complete. {len(new_df)} new records processed and saved to {output_path}"
    )
    log_success(f"Total records now: {len(final_df)}")


if __name__ == "__main__":
    main()
