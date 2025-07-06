# src/scripts/builders/build_player_features.py

import pandas as pd
from rapidfuzz import process
from collections import defaultdict
from scripts.utils.logger import log_info, log_success, log_warning
from scripts.utils.schema import normalize_columns

def find_fuzzy_match(name: str, choices: list[str], score_cutoff: int = 90) -> str | None:
    """Finds the best fuzzy match for a name from a list of choices."""
    if not name or pd.isna(name):
        return None
    match = process.extractOne(name, choices, score_cutoff=score_cutoff)
    return match[0] if match else None

def calculate_historical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rolling win pct, surface win pct, and H2H stats in a single,
    memory-efficient pass, ensuring data alignment.
    """
    log_info("Calculating all historical stats in a unified, memory-efficient pass...")
    
    # Sort by date to process matches chronologically and preserve original index
    df = df.sort_values(by='tourney_date').reset_index(drop=True)

    # Data structures to hold player histories
    player_matches = defaultdict(list)
    h2h_records = defaultdict(lambda: defaultdict(int))
    
    # List to store the results for each row
    results = []

    for index, row in df.iterrows():
        p1 = row['winner_name']
        p2 = row['loser_name']
        surface = row['surface']
        current_date = row['tourney_date']
        date_year_ago = current_date - pd.Timedelta(days=365)
        
        # --- Get H2H Stats (before this match) ---
        matchup_key = tuple(sorted((p1, p2)))
        p1_h2h_wins = h2h_records[matchup_key].get(p1, 0)
        p2_h2h_wins = h2h_records[matchup_key].get(p2, 0)

        # --- Get Historical Stats for P1 (winner) ---
        p1_history = [m for m in player_matches[p1] if m['date'] < current_date]
        p1_rolling_history = [m for m in p1_history if m['date'] >= date_year_ago]
        p1_surface_history = [m for m in p1_history if m['surface'] == surface]
        
        p1_rolling_win_pct = sum(m['win'] for m in p1_rolling_history) / len(p1_rolling_history) if p1_rolling_history else 0
        p1_surface_win_pct = sum(m['win'] for m in p1_surface_history) / len(p1_surface_history) if p1_surface_history else 0

        # --- Get Historical Stats for P2 (loser) ---
        p2_history = [m for m in player_matches[p2] if m['date'] < current_date]
        p2_rolling_history = [m for m in p2_history if m['date'] >= date_year_ago]
        p2_surface_history = [m for m in p2_history if m['surface'] == surface]

        p2_rolling_win_pct = sum(m['win'] for m in p2_rolling_history) / len(p2_rolling_history) if p2_rolling_history else 0
        p2_surface_win_pct = sum(m['win'] for m in p2_surface_history) / len(p2_surface_history) if p2_surface_history else 0

        # Append results for this match
        results.append({
            'p1_rolling_win_pct': p1_rolling_win_pct,
            'p2_rolling_win_pct': p2_rolling_win_pct,
            'p1_surface_win_pct': p1_surface_win_pct,
            'p2_surface_win_pct': p2_surface_win_pct,
            'p1_h2h_wins': p1_h2h_wins,
            'p2_h2h_wins': p2_h2h_wins,
        })

        # --- Update histories for future matches ---
        h2h_records[matchup_key][p1] += 1
        player_matches[p1].append({'date': current_date, 'win': 1, 'surface': surface})
        player_matches[p2].append({'date': current_date, 'win': 0, 'surface': surface})

    # Create a DataFrame from the results and concatenate it with the original
    stats_df = pd.DataFrame(results)
    df = pd.concat([df, stats_df], axis=1)
    
    return df


def build_player_features(sackmann_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to build player-centric features from historical data.
    """
    log_info("Building player features with canonical name matching...")
    
    # --- Memory Optimization ---
    for col in ['winner_name', 'loser_name', 'surface']:
        if col in sackmann_df.columns:
            sackmann_df[col] = sackmann_df[col].astype('category')
    if 'runner_name' in snapshots_df.columns:
        snapshots_df['runner_name'] = snapshots_df['runner_name'].astype('category')
    
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
        log_warning(f"Dropped {original_rows - len(sackmann_df)} rows due to missing name matches.")

    sackmann_df.drop(columns=['winner_name', 'loser_name'], inplace=True)
    sackmann_df.rename(columns={'winner_name_clean': 'winner_name', 'loser_name_clean': 'loser_name'}, inplace=True)

    sackmann_df['tourney_date'] = pd.to_datetime(sackmann_df['tourney_date'], format='%Y%m%d')

    features_df = calculate_historical_stats(sackmann_df)
    
    log_success("Successfully built all player features.")
    
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
        'p1_h2h_wins',
        'p2_h2h_wins',
    ]
    
    # This loop now correctly fills NaNs only for numeric columns.
    for col in final_cols:
        if col not in features_df.columns:
            if col in ['player_1', 'player_2', 'surface']:
                features_df[col] = None
                features_df[col] = features_df[col].astype('category')
            else:
                features_df[col] = 0.0
        else:
            if features_df[col].dtype.name != 'category':
                features_df[col] = features_df[col].fillna(0)
    
    return features_df[final_cols]


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Build player features from historical Sackmann data")
    parser.add_argument("--sackmann_csv", required=True)
    parser.add_argument("--snapshots_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    df_sackmann = pd.read_csv(args.sackmann_csv, low_memory=False)
    df_snapshots = pd.read_csv(args.snapshots_csv)
    features = build_player_features(df_sackmann, df_snapshots)
    features.to_csv(args.output_csv, index=False)
    log_info(f"Player features saved to {args.output_csv}")

if __name__ == "__main__":
    main_cli()