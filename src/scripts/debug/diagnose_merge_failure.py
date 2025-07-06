# src/scripts/debug/diagnose_merge_failure.py

import pandas as pd
from pathlib import Path
import sys
from rapidfuzz import process, fuzz
import yaml

# Add project root to path to allow imports from scripts
sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.utils.logger import setup_logging, log_info, log_error, log_warning, log_success

def find_fuzzy_match(name: str, choices: list[str], score_cutoff: int = 85):
    """Finds the best fuzzy match for a name from a list of choices."""
    if not name or pd.isna(name):
        return None, 0
    match = process.extractOne(name, choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
    return (match[0], match[1]) if match else (None, 0)

def get_tournament_config(label: str) -> dict | None:
    """Loads the main config and finds the specific configuration for a label."""
    config_path = Path("configs/tournaments_2024.yaml")
    if not config_path.exists():
        log_error(f"Main config not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    for tournament in full_config.get('tournaments', []):
        if tournament.get('label') == label:
            defaults = full_config.get('defaults', {})
            return {**defaults, **tournament}
    
    log_error(f"Could not find configuration for label '{label}' in {config_path}")
    return None

def diagnose_merge(label: str):
    """
    Loads a tournament's data and performs a deep diagnosis on the merge step.
    """
    setup_logging()
    log_info(f"--- Running Deep Merge Diagnostic for label: {label} ---")
    
    # --- Step 1: Get paths from the official config file ---
    tourn_config = get_tournament_config(label)
    if not tourn_config:
        return

    processed_dir = Path("data/processed")
    matches_path = processed_dir / f"{label}_matches_with_ids.csv"
    sackmann_path = Path(tourn_config.get("sackmann_csv"))
    snapshots_path = Path(tourn_config.get("snapshots_csv"))
    alias_path = Path(tourn_config.get("alias_csv"))
    
    for p in [sackmann_path, snapshots_path, alias_path]:
        if not p.exists():
            log_error(f"Required file not found: {p}")
            return
            
    if not matches_path.exists():
        log_error(f"File not found: {matches_path}. This means the 'ids' stage failed.")
        return

    try:
        # --- Load data ---
        matches_df = pd.read_csv(matches_path)
        sackmann_df = pd.read_csv(sackmann_path)
        alias_df = pd.read_csv(alias_path)

        # --- Pre-process Data ---
        log_info("Replicating pre-processing steps...")
        
        alias_map = alias_df.set_index('sackmann_name')['betfair_name'].to_dict()
        sackmann_df['winner_name_std'] = sackmann_df['winner_name'].map(alias_map).fillna(sackmann_df['winner_name'])
        sackmann_df['loser_name_std'] = sackmann_df['loser_name'].map(alias_map).fillna(sackmann_df['loser_name'])
        sackmann_df['tourney_date'] = pd.to_datetime(sackmann_df['tourney_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        historical_data = sackmann_df[['tourney_date', 'winner_name_std', 'loser_name_std']].copy()
        
        betting_data = matches_df[['tourney_date', 'player_1', 'player_2']].copy()
        betting_data.rename(columns={'tourney_date': 'match_date'}, inplace=True)
        
        # --- Deep Diagnosis ---
        log_info("Starting deep diagnosis of merge keys...")
        
        historical_names = set(historical_data['winner_name_std']).union(set(historical_data['loser_name_std']))
        
        unmatched_count = 0
        for index, row in betting_data.iterrows():
            p1_betting, p2_betting = row['player_1'], row['player_2']
            
            # Find the best possible match for each player
            p1_match, p1_score = find_fuzzy_match(p1_betting, list(historical_names))
            p2_match, p2_score = find_fuzzy_match(p2_betting, list(historical_names))
            
            # Check if a match exists in the historical data on the same day
            found_match = False
            if p1_match and p2_match:
                possible_matches = historical_data[
                    (historical_data['tourney_date'] == row['match_date']) &
                    (
                        ((historical_data['winner_name_std'] == p1_match) & (historical_data['loser_name_std'] == p2_match)) |
                        ((historical_data['winner_name_std'] == p2_match) & (historical_data['loser_name_std'] == p1_match))
                    )
                ]
                if not possible_matches.empty:
                    found_match = True
            
            if not found_match and unmatched_count < 5: # Limit to the first 5 for readability
                log_warning(f"\n--- Found Unmatched Betting Match #{unmatched_count + 1} ---")
                print(f"  Date: {row['match_date']}")
                print(f"  Betting Players: ('{p1_betting}', '{p2_betting}')")
                print(f"  Best fuzzy match for '{p1_betting}': '{p1_match}' (Score: {p1_score:.2f})")
                print(f"  Best fuzzy match for '{p2_betting}': '{p2_match}' (Score: {p2_score:.2f})")
                if p1_match and p2_match:
                     print(f"  Corresponding match in historical data on this day: Not Found")
                else:
                    print(f"  Reason: One or both players could not be confidently matched to any historical player.")
                unmatched_count += 1
                
        if unmatched_count == 0:
            log_success("SUCCESS! All betting matches have a corresponding entry in the historical data.")
        else:
            log_error(f"DIAGNOSIS COMPLETE: Found {unmatched_count} or more unmatched betting records.")
            log_info("Check the examples above to identify the names that need to be added to your 'player_aliases.csv' file.")

    except Exception as e:
        log_error(f"An exception occurred during the diagnostic run: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose feature merge issues for a specific tournament.")
    parser.add_argument("--label", required=True, help="The label of the tournament to diagnose (e.g., 'indianwells_2024_atp').")
    args = parser.parse_args()
    
    diagnose_merge(args.label)