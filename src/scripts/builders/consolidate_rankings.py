# src/scripts/builders/consolidate_rankings.py

import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.scripts.utils.config import load_config


def consolidate_rankings(input_glob: str, output_path: Path) -> None:
    """
    Consolidates multiple ranking CSVs into a single, sorted file.

    Args:
        input_glob (str): Glob pattern for the input CSV files.
        output_path (Path): The path to save the consolidated output CSV file.
    """
    files = glob.glob(input_glob)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_glob}")

    print(f"Found {len(files)} ranking files to consolidate.")

    df_list = [
        pd.read_csv(f, low_memory=False)
        for f in tqdm(files, desc="Reading Ranking CSVs")
    ]
    df = pd.concat(df_list, ignore_index=True)

    df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d")
    df = df.sort_values(by="ranking_date").reset_index(drop=True)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully consolidated {len(df)} ranking rows into {output_path}")


def main() -> None:
    """Main CLI entrypoint."""
    config = load_config("config.yaml")
    paths = config["data_paths"]

    atp_glob = paths["raw_atp_rankings_glob"]
    wta_glob = paths.get("raw_wta_rankings_glob", "")

    all_files = glob.glob(atp_glob) + (glob.glob(wta_glob) if wta_glob else [])

    if all_files:
        parent_dir = Path(all_files[0]).parent
        full_glob_pattern = str(parent_dir / "*.csv")
        consolidate_rankings(full_glob_pattern, Path(paths["consolidated_rankings"]))


if __name__ == "__main__":
    main()
