# src/scripts/debug/test_feature_merge.py

import pandas as pd
from pathlib import Path
import sys

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
    
    # --- CONFIGURATION ---
    # We will test with the 'indianwells_2024_atp' data as it was failing before.
    # You can change this to any other failing label if needed.
    label = "indianwells_2024_atp"
    # --- END CONFIGURATION ---

    log_info(f"--- Starting Debug Test for label: {label} ---")
    
    # Define paths to the input files for the 'features' stage
    processed_dir = Path("data/processed")
    merged_matches_path = processed_dir / f"{label}_merged_matches.csv"
    player_features_path = processed_dir / f"{label}_player_features.csv"
    
    # Check if input files exist
    if not merged_matches_path.exists() or not player_features_path.exists():
        log_error("Input files not found! Please run the full pipeline at least once to generate them.")
        log_error(f"Missing: {merged_matches_path}")
        log_error(f"Missing: {player_features_path}")
        return

    try:
        # Load the inputs
        log_info(f"Loading merged matches from: {merged_matches_path}")
        df_merged = pd.read_csv(merged_matches_path)
        
        log_info(f"Loading player features from: {player_features_path}")
        df_player_features = pd.read_csv(player_features_path)
        
        # Run the isolated function
        log_info("--- Running build_odds_features function ---")
        features_df = build_odds_features(df_merged, df_player_features)
        log_info("--- Function execution finished ---")

        # Verification
        if features_df.empty:
            log_error("💔 TEST FAILED: The resulting DataFrame is empty.")
            return
            
        columns_to_check = [
            'p1_rolling_win_pct', 'p2_rolling_win_pct', 
            'p1_surface_win_pct', 'p2_surface_win_pct'
        ]
        
        if features_df[columns_to_check].isnull().all().all():
            log_error("💔 TEST FAILED: The new feature columns are all empty.")
        else:
            populated_pct = (features_df[columns_to_check].notna().all(axis=1).sum() / len(features_df)) * 100
            log_success(f"🎉 TEST PASSED: Feature columns are populated in {populated_pct:.2f}% of rows.")
            
        # Save a sample for inspection
        output_path = Path("data/debug_features_output.csv")
        features_df.to_csv(output_path, index=False)
        log_info(f"Saved debug output to {output_path}")

    except Exception as e:
        log_error(f"An exception occurred during the debug run: {e}", exc_info=True)

if __name__ == "__main__":
    run_debug()