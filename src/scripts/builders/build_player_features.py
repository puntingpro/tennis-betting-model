# src/scripts/builders/build_player_features.py

import pandas as pd
from scripts.utils.logger import log_info, log_success
from scripts.utils.schema import normalize_columns

def calculate_surface_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates historical surface win percentage for each player in a memory-efficient way."""
    log_info("Calculating surface-specific win percentages...")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    # Melt the DataFrame to have one player per row
    id_vars = ['tourney_date', 'surface', 'match_num']
    melted = df.melt(id_vars=id_vars, value_vars=['winner_name', 'loser_name'], 
                     var_name='result', value_name='player_name')
    melted['win'] = (melted['result'] == 'winner_name').astype(int)
    melted = melted.sort_values(by=['player_name', 'tourney_date'])

    # Calculate cumulative wins and matches played on each surface
    melted['surface_wins'] = melted.groupby(['player_name', 'surface'])['win'].cumsum()
    melted['surface_matches'] = melted.groupby(['player_name', 'surface']).cumcount() + 1
    
    # Shift the results to get stats *prior* to the current match
    melted['prev_surface_wins'] = melted.groupby(['player_name', 'surface'])['surface_wins'].shift(1).fillna(0)
    melted['prev_surface_matches'] = melted.groupby(['player_name', 'surface'])['surface_matches'].shift(1).fillna(0)
    
    # Calculate win percentage, handling division by zero
    melted['surface_win_pct'] = melted['prev_surface_wins'] / melted['prev_surface_matches']
    melted['surface_win_pct'] = melted['surface_win_pct'].fillna(0)

    # Merge stats back for player 1 (winner)
    df = df.merge(
        melted[melted['result'] == 'winner_name'][['tourney_date', 'match_num', 'player_name', 'surface_win_pct']],
        left_on=['tourney_date', 'match_num', 'winner_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'surface_win_pct': 'p1_surface_win_pct'}).drop(columns='player_name')

    # Merge stats back for player 2 (loser)
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
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    # Melt the DataFrame to have one player per row
    id_vars = ['tourney_date', 'match_num']
    melted = df.melt(id_vars=id_vars, value_vars=['winner_name', 'loser_name'], 
                     var_name='result', value_name='player_name')
    melted['win'] = (melted['result'] == 'winner_name').astype(int)
    melted = melted.set_index('tourney_date').sort_index()

    # Perform rolling calculation
    rolling_stats = melted.groupby('player_name')['win'].rolling(window='365D', closed='left').agg(['sum', 'count'])
    rolling_stats.columns = ['rolling_wins', 'rolling_matches']
    rolling_stats = rolling_stats.reset_index()
    
    rolling_stats['rolling_win_pct'] = (rolling_stats['rolling_wins'] / rolling_stats['rolling_matches']).fillna(0)

    # Merge rolling stats back into the melted DataFrame
    melted = melted.reset_index().merge(rolling_stats, on=['player_name', 'tourney_date'], how='left')
    
    # Merge stats back for player 1 (winner)
    df = df.merge(
        melted[melted['result'] == 'winner_name'][['tourney_date', 'match_num', 'player_name', 'rolling_win_pct']],
        left_on=['tourney_date', 'match_num', 'winner_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'rolling_win_pct': 'p1_rolling_win_pct'}).drop(columns='player_name')

    # Merge stats back for player 2 (loser)
    df = df.merge(
        melted[melted['result'] == 'loser_name'][['tourney_date', 'match_num', 'player_name', 'rolling_win_pct']],
        left_on=['tourney_date', 'match_num', 'loser_name'],
        right_on=['tourney_date', 'match_num', 'player_name'],
        how='left'
    ).rename(columns={'rolling_win_pct': 'p2_rolling_win_pct'}).drop(columns='player_name')

    return df

def build_player_features(sackmann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to build player-centric features from historical data.
    """
    sackmann_df = normalize_columns(sackmann_df.copy())
    
    features_df = calculate_rolling_win_pct(sackmann_df)
    features_df = calculate_surface_win_pct(features_df)
    
    log_success("Successfully built player features.")
    
    # --- UPDATED SECTION ---
    # 1. Rename columns first
    features_df = features_df.rename(columns={
        'winner_name': 'player_1',
        'loser_name': 'player_2'
    })
    
    # 2. Define final columns with the NEW names
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

    # 3. Ensure all columns exist, fill missing with 0, and select them
    for col in final_cols:
        if col not in features_df.columns:
            features_df[col] = 0
            
    return features_df[final_cols]
    # --- END UPDATED SECTION ---

def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Build player features from historical Sackmann data")
    parser.add_argument("--sackmann_csv", required=True, help="Path to the Sackmann CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path to save the generated features.")
    args = parser.parse_args()
    
    df = pd.read_csv(args.sackmann_csv)
    features = build_player_features(df)
    features.to_csv(args.output_csv, index=False)
    log_info(f"Player features saved to {args.output_csv}")

if __name__ == "__main__":
    main_cli()