# src/scripts/debug/verify_features.py

import pandas as pd
import glob
from pathlib import Path
import sys
import os

# This ensures the script can find the project's root for correct pathing
sys.path.append(str(Path(__file__).resolve().parents[3]))

def verify_feature_columns():
    """
    Checks all *_features.csv files to ensure the new feature columns are populated.
    """
    # Use an absolute path from the project root
    project_root = Path(__file__).resolve().parents[3]
    feature_files = glob.glob(str(project_root / "data/processed/*_features.csv"))

    if not feature_files:
        print("❌ No feature files found in 'data/processed/'.")
        print("Please ensure the pipeline has been run successfully.")
        return

    print(f"🔎 Found {len(feature_files)} feature files to verify...")

    all_files_ok = True
    columns_to_check = [
        'p1_rolling_win_pct', 'p2_rolling_win_pct',
        'p1_surface_win_pct', 'p2_surface_win_pct',
        'p1_h2h_wins', 'p2_h2h_wins'
    ]

    for file_path in sorted(feature_files):
        file_name = Path(file_path).name
        try:
            df = pd.read_csv(file_path)

            if not all(col in df.columns for col in columns_to_check):
                print(f" Mismatch in file: {file_name}")
                all_files_ok = False
                continue

            if df[columns_to_check].isnull().all().all():
                print(f" Empty columns in file: {file_name}")
                all_files_ok = False
            else:
                populated_pct = (df[columns_to_check].notna().all(axis=1).sum() / len(df)) * 100
                if populated_pct < 1:
                     print(f" Partially populated file (might be ok): {file_name} ({populated_pct:.2f}% of rows have data)")
                else:
                    print(f"✅ Successfully verified: {file_name} ({populated_pct:.2f}% populated)")

        except Exception as e:
            print(f"❌ Error reading or processing {file_name}: {e}")
            all_files_ok = False

    print("-" * 30)
    if all_files_ok:
        print("🎉 All feature files appear to be correctly populated!")
    else:
        print("💔 Some issues were found. Please review the logs above.")

if __name__ == "__main__":
    verify_feature_columns()