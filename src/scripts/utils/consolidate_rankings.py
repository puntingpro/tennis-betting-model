# src/scripts/utils/consolidate_rankings.py

import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def consolidate_rankings(input_glob: str, output_csv: Path):
    """
    Consolidates multiple ranking CSVs into a single, sorted file.
    """
    files = glob.glob(input_glob)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_glob}")

    print(f"Found {len(files)} ranking files to consolidate.")

    df_list = []
    for f in tqdm(files, desc="Reading ranking CSVs"):
        df_list.append(pd.read_csv(f))

    print("Concatenating and sorting ranking data...")
    df = pd.concat(df_list, ignore_index=True)

    # Convert ranking_date to datetime for correct sorting
    df['ranking_date'] = pd.to_datetime(df['ranking_date'], format='%Y%m%d')
    df = df.sort_values(by='ranking_date').reset_index(drop=True)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving consolidated rankings to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully consolidated {len(df)} ranking rows into {output_csv}")

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Consolidate multiple ranking CSVs into one master file.")
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern for the input ranking CSVs (e.g., 'data/tennis_atp/atp_rankings_*.csv')."
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to save the consolidated output CSV."
    )
    args = parser.parse_args()
    consolidate_rankings(args.input_glob, args.output_csv)

if __name__ == "__main__":
    main()