# src/scripts/utils/consolidate_features.py
import pandas as pd
import glob
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Consolidate feature CSVs into a single file.")
    parser.add_argument("--input_glob", required=True, help="Glob pattern for the input feature files.")
    parser.add_argument("--output_csv", required=True, help="Path to save the consolidated output file.")
    args = parser.parse_args()

    files = glob.glob(args.input_glob)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {args.input_glob}")

    print(f"Found {len(files)} feature files to consolidate.")
    
    df_list = [pd.read_csv(f) for f in files]
    consolidated_df = pd.concat(df_list, ignore_index=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    consolidated_df.to_csv(output_path, index=False)
    print(f"✅ Successfully consolidated features into {output_path}")

if __name__ == "__main__":
    main()