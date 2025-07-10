# src/scripts/utils/consolidate_data.py

import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.scripts.utils.config import load_config # Added import

def consolidate_data(input_glob: str, output_csv: Path) -> None:
    """
    Consolidates multiple CSV files into a single, chronologically sorted file.
    """
    all_files = glob.glob(input_glob)
    if not all_files:
        raise FileNotFoundError(f"No files found matching the pattern: {input_glob}")

    exclude_keywords = ["doubles", "futures", "qual", "amateur"]
    csv_files = [
        f for f in all_files 
        if not any(keyword in Path(f).stem for keyword in exclude_keywords)
    ]
    
    print(f"Found {len(all_files)} files in total. After filtering, processing {len(csv_files)} main tour singles files.")

    df_list = [pd.read_csv(f, low_memory=False) for f in tqdm(csv_files, desc="Reading Match CSVs")]
    consolidated_df = pd.concat(df_list, ignore_index=True)
    
    consolidated_df['tourney_date'] = pd.to_datetime(consolidated_df['tourney_date'], format='%Y%m%d', errors='coerce')
    consolidated_df.dropna(subset=['tourney_date'], inplace=True)
    consolidated_df = consolidated_df.sort_values(by='tourney_date').reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_csv, index=False)
    print(f"✅ Successfully consolidated {len(consolidated_df)} match rows into {output_csv}")

def main():
    """Main CLI entrypoint."""
    # This function is now designed to be called from the main CLI script
    config = load_config("config.yaml")
    paths = config['data_paths']
    
    # Consolidate ATP and WTA matches together
    atp_glob = paths['raw_atp_matches_glob']
    wta_glob = paths.get('raw_wta_matches_glob', '') # Use .get for optional WTA data
    
    # Combine glob patterns
    full_glob_pattern = f"{Path(atp_glob).parent}/*.csv"
    
    consolidate_data(full_glob_pattern, Path(paths['consolidated_matches']))

if __name__ == "__main__":
    main()