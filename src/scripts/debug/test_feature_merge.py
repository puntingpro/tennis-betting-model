# src/scripts/debug/test_feature_merge.py

import pandas as pd
from pathlib import Path
import sys
import traceback

# Add project root to path to allow imports from scripts
sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.pipeline.build_odds_features import build_odds_features
from scripts.utils.logger import setup_logging, log_info, log_success, log_error

def run_debug():
    """
    Loads one tournament's data and runs the build_odds_features function
    in isolation to debug the feature merging logic.
    """
    setup_logging()
    
    label = "indianwells_2024_atp"
    log_info(f"--- Starting Final Diagnostic Test for label: {label} ---")
    
    processed_dir = Path("data/processed")
    merged_matches_path = processed_dir / f"{label}_merged_matches.csv"
    player_features_path = processed_dir / f"{label}_player_features.csv"
    
    if not merged_matches_path.exists() or not player_features_path.exists():
        log_error("Input files not found! Please run the pipeline again to generate them.")
        return

    try:
        df_merged = pd.read_csv(merged_matches_path)
        df_player_features = pd.read_csv(player_features_path)
        
        # --- DIAGNOSTIC PRINTING ---
        log_info("--- Data Samples Before Merge ---")
        
        # Prepare dates for consistent comparison
        df_merged['tourney_date_str'] = pd.to_datetime(df_merged['tourney_date']).dt.strftime('%Y-%m-%d')
        df_player_features['tourney_date_str'] = pd.to_datetime(df_player_features['tourney_date']).dt.strftime('%Y-%m-%d')
        
        # Define the keys we will use to merge
        merge_keys = ['tourney_date_str', 'player_1', 'surface']
        
        print("\n--- Sample from Merged Matches (Left DataFrame) ---")
        print(f"Columns to be used for merge key: {merge_keys}")
        print(df_merged[merge_keys].head(10))
        
        print("\n--- Sample from Player Features (Right DataFrame) ---")
        print(f"Columns to be used for merge key: {merge_keys}")
        print(df_player_features[merge_keys].head(10))
        
        log_info("--- End of Data Samples ---")
        # --- END DIAGNOSTIC ---

        log_info("--- Running build_odds_features function ---")
        features_df = build_odds_features(df_merged, df_player_features)
        log_info("--- Function execution finished ---")

        if features_df.empty or features_df[['p1_rolling_win_pct', 'p2_rolling_win_pct']].isnull().all().all():
            log_error("💔 DIAGNOSIS CONFIRMED: The new feature columns are empty because the merge keys do not match between the two files.")
            log_info("Please compare the player names, dates, and surfaces in the samples printed above to see the mismatch.")
        else:
            log_success("🎉 TEST PASSED: Features were successfully merged!")
            
        output_path = Path("data/debug_features_output.csv")
        features_df.to_csv(output_path, index=False)
        log_info(f"Saved debug output to {output_path}")

    except Exception as e:
        log_error(f"An exception occurred during the debug run: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()