# src/scripts/utils/consolidate_rankings.py

import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.scripts.utils.config import load_config # Added import

def consolidate_rankings(input_glob: str, output_csv: Path):
    """
    Consolidates multiple ranking CSVs into a single, sorted file.
    """
    files = glob.glob(input_glob)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_glob}")

    print(f"Found {len(files)} ranking files to consolidate.")

    df_list = [pd.read_csv(f) for f in tqdm(files, desc="Reading Ranking CSVs")]
    df = pd.concat(df_list, ignore_index=True)

    df['ranking_date'] = pd.to_datetime(df['ranking_date'], format='%Y%m%d')
    df = df.sort_values(by='ranking_date').reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Successfully consolidated {len(df)} ranking rows into {output_csv}")

def main():
    """Main CLI entrypoint."""
    config = load_config("config.yaml")
    paths = config['data_paths']

    # Consolidate ATP and WTA rankings
    atp_glob = paths['raw_atp_rankings_glob']
    wta_glob = paths.get('raw_wta_rankings_glob', '')
    
    full_glob_pattern = f"{Path(atp_glob).parent}/*.csv"

    consolidate_rankings(full_glob_pattern, Path(paths['consolidated_rankings']))

if __name__ == "__main__":
    main()