# src/scripts/builders/build_player_features.py

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a historical feature set for each player for each match.
    """
    # Initialize data stores
    player_stats = defaultdict(lambda: {
        'matches': 0,
        'wins': 0,
        'surface_matches': defaultdict(int),
        'surface_wins': defaultdict(int),
    })
    h2h_stats = defaultdict(lambda: {'p1_wins': 0, 'p2_wins': 0})
    
    features_list = []

    # Iterate over each match to calculate historical stats
    for row in tqdm(df.itertuples(), total=len(df), desc="Building Features"):
        winner_id, loser_id = row.winner_id, row.loser_id
        p1_id, p2_id = sorted((winner_id, loser_id))
        surface = row.surface

        h2h_record = h2h_stats[(p1_id, p2_id)]
        
        winner_current_stats = player_stats[winner_id]
        winner_surface_matches = winner_current_stats['surface_matches'][surface]
        winner_surface_wins = winner_current_stats['surface_wins'][surface]

        loser_current_stats = player_stats[loser_id]
        loser_surface_matches = loser_current_stats['surface_matches'][surface]
        loser_surface_wins = loser_current_stats['surface_wins'][surface]

        features_list.append({
            'match_id': row.match_id,
            'player_id': winner_id,
            'opponent_id': loser_id,
            'h2h_wins': h2h_record['p1_wins'] if winner_id == p1_id else h2h_record['p2_wins'],
            'h2h_losses': h2h_record['p2_wins'] if winner_id == p1_id else h2h_record['p1_wins'],
            'overall_win_rate': (winner_current_stats['wins'] / winner_current_stats['matches']) if winner_current_stats['matches'] > 0 else 0,
            'surface_win_rate': (winner_surface_wins / winner_surface_matches) if winner_surface_matches > 0 else 0,
        })
        
        features_list.append({
            'match_id': row.match_id,
            'player_id': loser_id,
            'opponent_id': winner_id,
            'h2h_wins': h2h_record['p2_wins'] if loser_id == p1_id else h2h_record['p1_wins'],
            'h2h_losses': h2h_record['p1_wins'] if loser_id == p1_id else h2h_record['p2_wins'],
            'overall_win_rate': (loser_current_stats['wins'] / loser_current_stats['matches']) if loser_current_stats['matches'] > 0 else 0,
            'surface_win_rate': (loser_surface_wins / loser_surface_matches) if loser_surface_matches > 0 else 0,
        })

        if winner_id == p1_id:
            h2h_stats[(p1_id, p2_id)]['p1_wins'] += 1
        else:
            h2h_stats[(p1_id, p2_id)]['p2_wins'] += 1

        player_stats[winner_id]['matches'] += 1
        player_stats[winner_id]['wins'] += 1
        player_stats[winner_id]['surface_matches'][surface] += 1
        player_stats[winner_id]['surface_wins'][surface] += 1

        player_stats[loser_id]['matches'] += 1
        player_stats[loser_id]['surface_matches'][surface] += 1
        
    return pd.DataFrame(features_list)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Build historical player features from Sackmann data."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to the consolidated Sackmann CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to save the generated player features CSV.",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # --- Preprocessing ---
    # Convert tourney_date to datetime - this is now handled by the consolidate script
    # df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d') # THIS LINE IS REMOVED
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    df['match_id'] = df['tourney_id'] + '-' + df['match_num'].astype(str)

    print("Building historical player features...")
    features_df = build_player_features(df)
    
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving features to {args.output_csv}...")
    features_df.to_csv(args.output_csv, index=False)
    
    print(f"✅ Successfully created player feature library at {args.output_csv}")
    print(f"Generated {len(features_df)} feature rows.")


if __name__ == "__main__":
    main()