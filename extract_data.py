import tarfile
import json
import pandas as pd
import bz2
import sys
import os
from concurrent.futures import ProcessPoolExecutor


def process_tar_member(member_tuple):
    """
    This function processes a single member of the tar archive.
    It's designed to be run in a separate process to leverage multiple CPU cores.
    """
    tar_path, member_info = member_tuple
    market_data = []

    with tarfile.open(tar_path, "r") as tar:
        file_obj = tar.extractfile(member_info)
        if file_obj:
            with bz2.open(file_obj, "rt") as bz2f:
                for line in bz2f:
                    try:
                        change = json.loads(line)
                        if "mc" not in change:
                            continue

                        for market_change in change["mc"]:
                            market_def = market_change.get("marketDefinition", {})

                            # --- FIX: Filter out non-singles matches and non-MATCH_ODDS markets ---
                            if (
                                "/" in market_def.get("eventName", "")
                                or market_def.get("marketType") != "MATCH_ODDS"
                            ):
                                continue

                            # --- FIX: Extract best available back price and volume from 'atb' ---
                            runner_changes = market_change.get("rc", [])
                            for runner_change in runner_changes:
                                best_back_price = None
                                best_back_volume = None

                                available_to_back = runner_change.get("atb", [])
                                if available_to_back:
                                    best_back_price = available_to_back[0][0]
                                    best_back_volume = available_to_back[0][1]

                                # Find the corresponding runner name from the market definition
                                runner_id = runner_change.get("id")
                                runner_info = next(
                                    (
                                        r
                                        for r in market_def.get("runners", [])
                                        if r.get("id") == runner_id
                                    ),
                                    None,
                                )

                                if runner_info:
                                    market_data.append(
                                        {
                                            "market_id": market_change.get("id"),
                                            "event_name": market_def.get("eventName"),
                                            "market_start_time": market_def.get(
                                                "marketTime"
                                            ),
                                            "market_status": market_def.get("status"),
                                            "in_play": market_def.get("inPlay"),
                                            "runner_id": runner_id,
                                            "runner_name": runner_info.get("name"),
                                            "runner_status": runner_info.get("status"),
                                            "best_back_price": best_back_price,
                                            "best_back_volume": best_back_volume,
                                            "pt": change.get("pt"),  # Published Time
                                        }
                                    )
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    return market_data


def main(tar_path):
    """
    Main function to extract data from the tar archive and save it to a CSV file.
    """
    if not os.path.exists(tar_path):
        print(f"Error: File not found at {tar_path}")
        return

    print(f"Starting data extraction from {tar_path}...")

    with tarfile.open(tar_path, "r") as tar:
        members = [(tar_path, member) for member in tar.getmembers() if member.isfile()]

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_tar_member, members)

    all_market_data = [item for sublist in results for item in sublist]

    if not all_market_data:
        print("No valid singles MATCH_ODDS market data found in the archive.")
        return

    df = pd.DataFrame(all_market_data)
    df.dropna(
        subset=["best_back_price"], inplace=True
    )  # Remove records where price wasn't available

    df["pt"] = pd.to_datetime(df["pt"], unit="ms")
    df.sort_values(by=["market_id", "pt"], inplace=True)

    output_path = "tennis_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Data extraction complete. {len(df)} records saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_data.py <path_to_tar_file>")
    else:
        main(sys.argv[1])
