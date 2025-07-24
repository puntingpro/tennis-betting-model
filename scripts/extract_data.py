import tarfile
import orjson as json  # MODIFICATION: Use the much faster orjson library
import pandas as pd
import bz2
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm


def process_tar_file(tar_path):
    """
    Processes a single tar archive and saves its results to a temporary parquet file.
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
                                    market_type = market_def.get("marketType")
                                    if market_type not in [
                                        "SET_WINNER",
                                        "SET_1_WINNER",
                                        "SET_2_WINNER",
                                    ]:
                                        continue
                                    if "/" in market_def.get("eventName", ""):
                                        continue
                                    for runner_change in market_change.get("rc", []):
                                        atb = runner_change.get("atb", [])
                                        if atb:
                                            runner_id = runner_change.get("id")
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
                                                        "market_id": market_change.get(
                                                            "id"
                                                        ),
                                                        "event_name": market_def.get(
                                                            "eventName"
                                                        ),
                                                        "market_type": market_type,
                                                        "market_start_time": market_def.get(
                                                            "marketTime"
                                                        ),
                                                        "runner_id": runner_id,
                                                        "runner_name": runner_info.get(
                                                            "name"
                                                        ),
                                                        "runner_status": runner_info.get(
                                                            "status"
                                                        ),
                                                        "best_back_price": atb[0][0],
                                                        "pt": change.get("pt"),
                                                    }
                                                )
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
    except tarfile.ReadError:
        print(f"Warning: Could not read {tar_path}, it may be corrupted. Skipping.")

    # MODIFICATION: Write to a temporary file instead of returning a large list
    if market_data:
        temp_df = pd.DataFrame(market_data)
        temp_filename = f"temp_progress_{os.path.basename(tar_path)}.parquet"
        temp_df.to_parquet(temp_filename)
        return temp_filename
    return None


def main(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return
    tar_files = glob.glob(os.path.join(folder_path, "*.tar"))
    if not tar_files:
        print(f"No .tar files found in {folder_path}")
        return

    print(f"Found {len(tar_files)} tar files to process. Starting data extraction...")

    with ProcessPoolExecutor() as executor:
        temp_files = list(
            tqdm(
                executor.map(process_tar_file, tar_files),
                total=len(tar_files),
                desc="Processing TAR files",
            )
        )

    # MODIFICATION: Consolidate temporary files at the end
    print("\nConsolidating all processed data...")
    all_dfs = [pd.read_parquet(f) for f in temp_files if f is not None]

    if not all_dfs:
        print("No valid singles SET_WINNER market data found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df.dropna(subset=["best_back_price"], inplace=True)
    final_df["pt"] = pd.to_datetime(final_df["pt"], unit="ms")

    output_path = "tennis_set_data.csv"
    final_df.to_csv(output_path, index=False)
    print(
        f"Data consolidation complete. {len(final_df)} records saved to {output_path}"
    )

    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        if temp_file is not None:
            os.remove(temp_file)
    print("Cleanup complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_data.py <path_to_folder_with_tar_files>")
    else:
        main(sys.argv[1])
