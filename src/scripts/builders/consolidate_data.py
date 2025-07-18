# src/scripts/builders/consolidate_data.py

import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.scripts.utils.config import load_config


def consolidate_data(input_glob: str, output_path: Path) -> None:
    """
    Consolidates multiple match data CSVs into a single, chronologically sorted file.

    It filters out doubles, qualifiers, and amateur matches, but makes a specific
    exception to include the main-draw Challenger files.

    Args:
        input_glob (str): Glob pattern for the input CSV files.
        output_path (Path): The path to save the consolidated output CSV file.
    """
    all_files = glob.glob(input_glob)
    if not all_files:
        raise FileNotFoundError(f"No files found matching the pattern: {input_glob}")

    # --- BUG FIX: Added more specific logic to handle 'qual_chall' files correctly ---
    exclude_keywords = ["doubles", "qual", "amateur"]
    
    csv_files = []
    for f in all_files:
        filename_stem = Path(f).stem
        # Keep the file if it's a challenger file, even if 'qual' is in the name
        if "qual_chall" in filename_stem:
            csv_files.append(f)
            continue
        # Otherwise, exclude the file if it contains any of the exclusion keywords
        if not any(keyword in filename_stem for keyword in exclude_keywords):
            csv_files.append(f)
    # --- END FIX ---

    print(
        f"Found {len(all_files)} files. After filtering, processing {len(csv_files)} files."
    )

    df_list = [
        pd.read_csv(f, low_memory=False)
        for f in tqdm(csv_files, desc="Reading Match CSVs")
    ]
    consolidated_df = pd.concat(df_list, ignore_index=True)

    consolidated_df["tourney_date"] = pd.to_datetime(
        consolidated_df["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    consolidated_df.dropna(subset=["tourney_date"], inplace=True)
    consolidated_df = consolidated_df.sort_values(by="tourney_date").reset_index(
        drop=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)
    print(
        f"✅ Successfully consolidated {len(consolidated_df)} match rows into {output_path}"
    )


def main() -> None:
    """Main CLI entrypoint."""
    config = load_config("config.yaml")
    paths = config["data_paths"]

    atp_glob = paths["raw_atp_matches_glob"]
    wta_glob = paths.get("raw_wta_matches_glob", "")

    all_files = glob.glob(atp_glob) + (glob.glob(wta_glob) if wta_glob else [])

    if all_files:
        parent_dir = Path(all_files[0]).parent
        full_glob_pattern = str(parent_dir / "*.csv")
        consolidate_data(full_glob_pattern, Path(paths["consolidated_matches"]))


if __name__ == "__main__":
    main()