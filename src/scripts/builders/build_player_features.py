# src/scripts/builders/build_player_features.py

import pandas as pd
from rapidfuzz import process
from scripts.utils.logger import log_info, log_success
from scripts.utils.schema import normalize_columns

def find_fuzzy_match(name: str, choices: list[str], score_cutoff: int = 90) -> str | None:
    """Finds the best fuzzy match for a name from a list of choices."""
    if not name or pd.isna(name):
        return None
    match = process.extractOne(name, choices, score_cutoff=score_cutoff)
    return match[0] if match else None

def calculate_surface_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates historical surface win percentage for each player in a memory-efficient way."""
    log_info("Calculating surface-specific win percentages...")
    
    id_vars = ['tourney_date', 'surface', 'match_num']
    melted = df.melt(id_vars=id_vars, value_vars=['winner_name', 'loser_name'], 
                     var_name='result', value_name='player_name')
    melted['win'] = (melted['result'] == 'winner_name').astype(int)
    melted = melted.sort_values(by=['player_name', 'tourney_date'])

    melted['surface_wins'] = melted.groupby(['player_name', 'surface'])['win'].cumsum()
    melted['surface_matches'] = melted.groupby(['player_name', 'surface']).cumcount() + 1
    
    melted['prev_surface_wins'] = melted.groupby(['player_name', 'surface'])['surface_wins'].shift(1).fillna(0)
    melted['prev_surface_matches'] = melted.groupby(['player_name', 'surface'])['surface_matches'].shift(1).fillna(0)
    
    melted['surface_win_pct'] = (melted['prev_surface_wins'] / melted['prev_surface_matches']).fillna(0)

    df = df.merge(
        melted[melted['result'] == 'winner_name'][['tourney_date', 'match_num', 'player_name', 'surface_win_pct']],
        left_on=['tourney_date', 'match_num', 'winner_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'surface_win_pct': 'p1_surface_win_pct'}).drop(columns='player_name')

    df = df.merge(
        melted[melted['result'] == 'loser_name'][['tourney_date', 'match_num', 'player_name', 'surface_win_pct']],
        left_on=['tourney_date', 'match_num', 'loser_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'surface_win_pct': 'p2_surface_win_pct'}).drop(columns='player_name')

    return df

def calculate_rolling_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a 52-week rolling win percentage for each player in a memory-efficient way."""
    log_info("Calculating 52-week rolling win percentages...")
    
    id_vars = ['tourney_date', 'match_num']
    melted = df.melt(id_vars=id_vars, value_vars=['winner_name', 'loser_name'], 
                     var_name='result', value_name='player_name')
    melted['win'] = (melted['result'] == 'winner_name').astype(int)
    melted = melted.set_index('tourney_date').sort_index()

    rolling_stats = melted.groupby('player_name')['win'].rolling(window='365D', closed='left').agg(['sum', 'count'])
    rolling_stats.columns = ['rolling_wins', 'rolling_matches']
    rolling_stats = rolling_stats.reset_index()
    
    rolling_stats['rolling_win_pct'] = (rolling_stats['rolling_wins'] / rolling_stats['rolling_matches']).fillna(0)

    melted = melted.reset_index().merge(rolling_stats, on=['player_name', 'tourney_date'], how='left')
    
    df = df.merge(
        melted[melted['result'] == 'winner_name'][['tourney_date', 'match_num', 'player_name', 'rolling_win_pct']],
        left_on=['tourney_date', 'match_num', 'winner_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'rolling_win_pct': 'p1_rolling_win_pct'}).drop(columns='player_name')

    df = df.merge(
        melted[melted['result'] == 'loser_name'][['tourney_date', 'match_num', 'player_name', 'rolling_win_pct']],
        left_on=['tourney_date', 'match_num', 'loser_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'rolling_win_pct': 'p2_rolling_win_pct'}).drop(columns='player_name')

    return df

def build_player_features(sackmann_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to build player-centric features from historical data,
    using canonical player names from snapshots for matching.
    """
    log_info("Building player features with canonical name matching...")
    sackmann_df = normalize_columns(sackmann_df.copy())
    snapshots_df = normalize_columns(snapshots_df.copy())

    unique_runner_names = snapshots_df["runner_name"].dropna().unique().tolist()
    log_info(f"Found {len(unique_runner_names)} canonical runner names for matching.")

    log_info("Applying fuzzy matching to historical winner/loser names...")
    sackmann_df["winner_name_clean"] = sackmann_df["winner_name"].apply(
        lambda x: find_fuzzy_match(x, unique_runner_names)
    )
    sackmann_df["loser_name_clean"] = sackmann_df["loser_name"].apply(
        lambda x: find_fuzzy_match(x, unique_runner_names)
    )
    
    original_rows = len(sackmann_df)
    sackmann_df.dropna(subset=['winner_name_clean', 'loser_name_clean'], inplace=True)
    if original_rows > len(sackmann_df):
        log_info(f"Dropped {original_rows - len(sackmann_df)} rows due to missing name matches.")

    sackmann_df.drop(columns=['winner_name', 'loser_name'], inplace=True)
    sackmann_df.rename(columns={'winner_name_clean': 'winner_name', 'loser_name_clean': 'loser_name'}, inplace=True)

    # CRITICAL FIX: Convert date from YYYYMMDD format to datetime objects for calculations
    sackmann_df['tourney_date'] = pd.to_datetime(sackmann_df['tourney_date'], format='%Y%m%d')

    features_df = calculate_rolling_win_pct(sackmann_df)
    features_df = calculate_surface_win_pct(features_df)
    
    log_success("Successfully built player features.")
    
    # Convert date to string for the final CSV output
    features_df['tourney_date'] = features_df['tourney_date'].dt.strftime('%Y-%m-%d')
    
    features_df = features_df.rename(columns={
        'winner_name': 'player_1',
        'loser_name': 'player_2'
    })
    
    final_cols = [
        'tourney_date',
        'surface',
        'player_1',
        'player_2',
        'p1_rolling_win_pct',
        'p2_rolling_win_pct',
        'p1_surface_win_pct',
        'p2_surface_win_pct',
    ]
    
    for col in final_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    
    return features_df[final_cols].fillna(0)

def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Build player features from historical Sackmann data")
    parser.add_argument("--sackmann_csv", required=True)
    parser.add_argument("--snapshots_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    df_sackmann = pd.read_csv(args.sackmann_csv)
    df_snapshots = pd.read_csv(args.snapshots_csv)
    features = build_player_features(df_sackmann, df_snapshots)
    features.to_csv(args.output_csv, index=False)
    log_info(f"Player features saved to {args.output_csv}")

if __name__ == "__main__":
    main_cli()