# src/scripts/builders/consolidate_data.py

import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.scripts.utils.config import load_config

def consolidate_data(input_glob: str, output_path: Path) -> None:
    """
    Consolidates multiple match data CSVs into a single, chronologically sorted file.

    It filters out doubles, futures, qualifiers, and amateur matches.

    Args:
        input_glob (str): Glob pattern for the input CSV files.
        output_path (Path): The path to save the consolidated output CSV file.
    """
    all_files = glob.glob(input_glob)
    if not all_files:
        raise FileNotFoundError(f"No files found matching the pattern: {input_glob}")

    exclude_keywords = ["doubles", "futures", "qual", "amateur"]
    csv_files = [
        f for f in all_files 
        if not any(keyword in Path(f).stem for keyword in exclude_keywords)
    ]
    
    print(f"Found {len(all_files)} files. After filtering, processing {len(csv_files)} main tour singles files.")

    df_list = [pd.read_csv(f, low_memory=False) for f in tqdm(csv_files, desc="Reading Match CSVs")]
    consolidated_df = pd.concat(df_list, ignore_index=True)
    
    consolidated_df['tourney_date'] = pd.to_datetime(consolidated_df['tourney_date'], format='%Y%m%d', errors='coerce')
    consolidated_df.dropna(subset=['tourney_date'], inplace=True)
    consolidated_df = consolidated_df.sort_values(by='tourney_date').reset_index(drop=True)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)
    print(f"✅ Successfully consolidated {len(consolidated_df)} match rows into {output_path}")

def main() -> None:
    """Main CLI entrypoint."""
    config = load_config("config.yaml")
    paths = config['data_paths']
    
    atp_glob = paths['raw_atp_matches_glob']
    wta_glob = paths.get('raw_wta_matches_glob', '')
    
    # Combine glob patterns from both directories
    all_files = glob.glob(atp_glob) + (glob.glob(wta_glob) if wta_glob else [])
    
    # Create a unified glob pattern from the parent directory of the first file
    if all_files:
        parent_dir = Path(all_files[0]).parent
        full_glob_pattern = str(parent_dir / "*.csv")
        consolidate_data(full_glob_pattern, Path(paths['consolidated_matches']))

if __name__ == "__main__":
    main()