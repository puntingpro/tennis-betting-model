# src/scripts/utils/consolidate_data.py

import argparse
import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def consolidate_data(input_glob: str, output_csv: Path) -> None:
    """
    Consolidates multiple CSV files into a single, chronologically sorted file.
    """
    all_files = glob.glob(input_glob)
    if not all_files:
        raise FileNotFoundError(f"No files found matching the pattern: {input_glob}")

    # --- MODIFIED SECTION: Intelligent File Filtering ---
    # Define keywords for files to exclude to ensure we only get main tour singles.
    exclude_keywords = ["doubles", "futures", "qual", "amateur"]
    
    csv_files = [
        f for f in all_files 
        if not any(keyword in Path(f).stem for keyword in exclude_keywords)
    ]
    
    print(f"Found {len(all_files)} files in total. After filtering, processing {len(csv_files)} main tour singles files.")
    # --- END MODIFIED SECTION ---

    df_list = []
    for f in tqdm(csv_files, desc="Reading CSVs"):
        df_list.append(pd.read_csv(f, low_memory=False))

    print("Concatenating data...")
    consolidated_df = pd.concat(df_list, ignore_index=True)

    initial_rows = len(consolidated_df)
    print(f"Converting tourney_date and handling errors. Initial rows: {initial_rows}")
    
    consolidated_df['tourney_date'] = pd.to_datetime(
        consolidated_df['tourney_date'], format='%Y%m%d', errors='coerce'
    )
    
    consolidated_df.dropna(subset=['tourney_date'], inplace=True)
    
    final_rows = len(consolidated_df)
    if final_rows < initial_rows:
        print(f"Dropped {initial_rows - final_rows} rows due to invalid dates.")

    print("Sorting data by date...")
    consolidated_df = consolidated_df.sort_values(by='tourney_date').reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving consolidated file to {output_csv}...")
    consolidated_df.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully consolidated {len(consolidated_df)} rows into {output_csv}")

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Consolidate multiple Sackmann CSVs into one master file."
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern to find the input CSVs (e.g., 'data/tennis_atp/atp_matches_*.csv').",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to save the consolidated output CSV.",
    )
    args = parser.parse_args()

    consolidate_data(args.input_glob, args.output_csv)

if __name__ == "__main__":
    main()