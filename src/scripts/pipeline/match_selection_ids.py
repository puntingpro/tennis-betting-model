# src/scripts/pipeline/match_selection_ids.py

import pandas as pd
from scripts.utils.logger import log_info, log_warning, log_success
from scripts.utils.schema import enforce_schema, normalize_columns
from scripts.utils.selection import build_market_runner_map

def assign_selection_ids(
    matches_df: pd.DataFrame, 
    sackmann_df: pd.DataFrame, 
    snapshots_df: pd.DataFrame, 
    alias_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identifies players and merges historical data using a definitive alias map and a canonical matchup key.
    """
    log_info("Starting robust player and historical data assignment...")

    # --- Step 1: Normalize all dataframes ---
    matches_df = normalize_columns(matches_df)
    sackmann_df = normalize_columns(sackmann_df)
    snapshots_df = normalize_columns(snapshots_df)
    alias_df = normalize_columns(alias_df)

    # --- Step 2: Apply aliases to historical data ---
    if not alias_df.empty:
        alias_map = alias_df.set_index('sackmann_name')['betfair_name'].to_dict()
        sackmann_df['winner_name'] = sackmann_df['winner_name'].map(alias_map).fillna(sackmann_df['winner_name'])
        sackmann_df['loser_name'] = sackmann_df['loser_name'].map(alias_map).fillna(sackmann_df['loser_name'])

    # --- Step 3: Prepare historical data ---
    sackmann_df['tourney_date'] = pd.to_datetime(sackmann_df['tourney_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    
    # --- Step 4: Prepare betting data ---
    market_to_runners = snapshots_df.groupby('market_id')['runner_name'].unique().apply(list)
    pivoted_matches = matches_df[['match_id', 'timestamp']].drop_duplicates(subset=['match_id'])
    pivoted_matches['runners'] = pivoted_matches['match_id'].map(market_to_runners)
    pivoted_matches.dropna(subset=['runners'], inplace=True)
    pivoted_matches = pivoted_matches[pivoted_matches['runners'].apply(len) == 2]
    pivoted_matches[['player_1', 'player_2']] = pd.DataFrame(pivoted_matches['runners'].tolist(), index=pivoted_matches.index)
    pivoted_matches['match_date'] = pd.to_datetime(pivoted_matches['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')

    # --- Step 5: Create Canonical Matchup Keys ---
    def create_matchup_key(row, p1_col, p2_col):
        p1, p2 = row[p1_col], row[p2_col]
        return tuple(sorted((p1, p2))) if pd.notna(p1) and pd.notna(p2) else None

    pivoted_matches['matchup_key'] = pivoted_matches.apply(create_matchup_key, p1_col='player_1', p2_col='player_2', axis=1)
    sackmann_df['matchup_key'] = sackmann_df.apply(create_matchup_key, p1_col='winner_name', p2_col='loser_name', axis=1)
    
    # --- Step 6: Perform the final, robust merge ---
    merged_df = pd.merge(
        pivoted_matches,
        sackmann_df,
        left_on=['match_date', 'matchup_key'],
        right_on=['tourney_date', 'matchup_key'],
        how='inner' # Use inner merge to keep only exact matches
    )

    if merged_df.empty:
        log_warning("No matches could be aligned with historical data after robust merge.")
        return enforce_schema(pd.DataFrame(), "matches_with_ids")

    log_success(f"Successfully merged data, resulting in {len(merged_df)} matched records.")

    # --- Step 7: Finalize the DataFrame ---
    merged_df.rename(columns={'winner_name': 'winner'}, inplace=True)
    
    result_df = matches_df.merge(
        merged_df[['match_id', 'player_1', 'player_2', 'winner', 'surface', 'tourney_date']],
        on='match_id',
        how='inner'
    )
    
    market_map = build_market_runner_map(snapshots_df)
    result_df["selection_id_1"] = result_df.apply(lambda r: market_map.get(r["match_id"], {}).get(r["player_1"]), axis=1)
    result_df["selection_id_2"] = result_df.apply(lambda r: market_map.get(r["match_id"], {}).get(r["player_2"]), axis=1)

    return enforce_schema(result_df, "matches_with_ids")

def main_cli(args=None):
    log_info("This script is primarily used as a module within the main pipeline.")

if __name__ == "__main__":
    main_cli()