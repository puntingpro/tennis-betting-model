# src/scripts/pipeline/match_selection_ids.py

import pandas as pd
from scripts.utils.logger import log_info, log_warning, log_success
from scripts.utils.schema import enforce_schema, normalize_columns
from scripts.utils.selection import build_market_runner_map

def _create_matchup_key(row, p1_col, p2_col):
    """Helper to safely create a matchup key, returning None if names are invalid."""
    p1, p2 = row.get(p1_col), row.get(p2_col)
    if isinstance(p1, str) and isinstance(p2, str):
        return tuple(sorted((p1, p2)))
    return None

def assign_selection_ids(
    matches_df: pd.DataFrame,
    sackmann_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    alias_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final, most robust version to assign selection IDs by ensuring data integrity
    before any merge operations.
    """
    log_info("Starting final robust player and historical data assignment...")

    # --- Step 1: Normalize all dataframes ---
    matches_df = normalize_columns(matches_df.copy())
    sackmann_df = normalize_columns(sackmann_df.copy())
    snapshots_df = normalize_columns(snapshots_df.copy())
    alias_df = normalize_columns(alias_df.copy())

    # --- Step 2: Prepare Historical Data ---
    log_info("Preparing historical data...")
    alias_map = alias_df.set_index("sackmann_name")["betfair_name"].to_dict()
    sackmann_df["winner_name"] = sackmann_df["winner_name"].map(alias_map)
    sackmann_df["loser_name"] = sackmann_df["loser_name"].map(alias_map)
    
    # **GUARANTEED FIX: Use the safe key creation function and drop invalid rows**
    sackmann_df['matchup_key'] = sackmann_df.apply(_create_matchup_key, p1_col='winner_name', p2_col='loser_name', axis=1)
    sackmann_df.dropna(subset=['matchup_key'], inplace=True)
    
    sackmann_df["tourney_date"] = pd.to_datetime(sackmann_df["tourney_date"], format="%Y%m%d")

    # --- Step 3: Prepare Betting Data ---
    log_info("Preparing betting data...")
    market_to_runners = snapshots_df.groupby('market_id')['runner_name'].unique().apply(list)
    
    # Work on a copy to avoid SettingWithCopyWarning
    pivoted_matches = matches_df[['match_id', 'timestamp']].drop_duplicates().copy()
    pivoted_matches['runners'] = pivoted_matches['match_id'].map(market_to_runners)
    pivoted_matches.dropna(subset=['runners'], inplace=True)
    pivoted_matches = pivoted_matches[pivoted_matches['runners'].apply(lambda x: isinstance(x, list) and len(x) == 2)]

    if pivoted_matches.empty:
        log_warning("No valid two-runner markets found. Cannot proceed.")
        return enforce_schema(pd.DataFrame(), "matches_with_ids")

    pivoted_matches[['player_1', 'player_2']] = pd.DataFrame(pivoted_matches['runners'].tolist(), index=pivoted_matches.index)
    
    # **GUARANTEED FIX: Use the safe key creation function and drop invalid rows**
    pivoted_matches['matchup_key'] = pivoted_matches.apply(_create_matchup_key, p1_col='player_1', p2_col='player_2', axis=1)
    pivoted_matches.dropna(subset=['matchup_key'], inplace=True)
    
    pivoted_matches['match_date'] = pd.to_datetime(pivoted_matches['timestamp'], unit='ms')

    # --- Step 4: Perform the Final Merge ---
    log_info("Performing final robust merge...")
    
    merged_df = pd.merge_asof(
        pivoted_matches.sort_values('match_date'),
        sackmann_df.sort_values('tourney_date'),
        left_on='match_date',
        right_on='tourney_date',
        by='matchup_key',
        direction='nearest',
        tolerance=pd.Timedelta(days=2)
    )

    merged_df.dropna(subset=['tourney_id'], inplace=True)

    if merged_df.empty:
        log_warning("No matches could be aligned. Check aliases and date ranges.")
        return enforce_schema(pd.DataFrame(), "matches_with_ids")

    log_success(f"Successfully merged {len(merged_df)} records.")

    # --- Step 5: Finalize the DataFrame ---
    result_df = matches_df.merge(merged_df, on="match_id", how="inner", suffixes=('', '_merged'))
    result_df.rename(columns={'winner_name': 'winner'}, inplace=True)
    
    market_map = build_market_runner_map(snapshots_df)
    result_df["selection_id_1"] = result_df.apply(lambda r: market_map.get(r["match_id"], {}).get(r["player_1"]), axis=1)
    result_df["selection_id_2"] = result_df.apply(lambda r: market_map.get(r["match_id"], {}).get(r["player_2"]), axis=1)
    result_df['tourney_date'] = result_df['tourney_date'].dt.strftime('%Y-%m-%d')

    return enforce_schema(result_df, "matches_with_ids")