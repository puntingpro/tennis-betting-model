# src/scripts/builders/build_player_features.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, deque

def get_most_recent_ranking(df_rankings, player_id, date):
    player_rankings = df_rankings[df_rankings['player'] == player_id]
    last_ranking_idx = player_rankings['ranking_date'].searchsorted(date, side='right') - 1
    if last_ranking_idx >= 0:
        return player_rankings.iloc[last_ranking_idx]['rank']
    return np.nan

def build_advanced_player_features(df_matches, df_rankings, df_players):
    """
    Builds an advanced historical feature set for each player, now including
    recent form (rolling win percentage).
    """
    player_info = df_players.set_index('player_id').to_dict('index')
    df_matches = df_matches.sort_values(by='tourney_date').reset_index(drop=True)
    df_rankings = df_rankings.sort_values(by='ranking_date').reset_index(drop=True)

    # --- MODIFIED: More complex stats storage, including recent form ---
    player_stats = defaultdict(lambda: {
        'wins': 0, 'matches': 0,
        'surface_wins': defaultdict(int),
        'surface_matches': defaultdict(int),
        'last_10_results': deque(maxlen=10) # Use a deque for efficient rolling window
    })
    h2h_stats = {}
    
    features_list = []

    for row in tqdm(df_matches.itertuples(), total=len(df_matches), desc="Building Form-Aware Features"):
        winner_id, loser_id = int(row.winner_id), int(row.loser_id)
        match_date = row.tourney_date
        surface = row.surface if pd.notna(row.surface) else 'Unknown'
        tourney_name = row.tourney_name

        winner_info = player_info.get(winner_id, {})
        loser_info = player_info.get(loser_id, {})

        winner_rank = get_most_recent_ranking(df_rankings, winner_id, match_date)
        loser_rank = get_most_recent_ranking(df_rankings, loser_id, match_date)

        winner_hist = player_stats[winner_id]
        loser_hist = player_stats[loser_id]
        
        h2h_key = tuple(sorted((winner_id, loser_id)))
        h2h_hist = h2h_stats.get(h2h_key, {'p1_wins': 0, 'p2_wins': 0, 'total': 0})

        # --- Create Matchup Row with Form Features ---
        p1_id, p2_id = (winner_id, loser_id)

        # Get stats *before* the current match
        p1_form = sum(winner_hist['last_10_results']) / len(winner_hist['last_10_results']) if winner_hist['last_10_results'] else 0
        p2_form = sum(loser_hist['last_10_results']) / len(loser_hist['last_10_results']) if loser_hist['last_10_results'] else 0

        features_list.append({
            'match_id': row.match_id, 'tourney_date': match_date, 'tourney_name': tourney_name,
            'p1_id': p1_id, 'p2_id': p2_id, 'winner': 1,
            'p1_rank': winner_rank, 'p2_rank': loser_rank,
            'rank_diff': winner_rank - loser_rank if pd.notna(winner_rank) and pd.notna(loser_rank) else 0,
            'p1_height': winner_info.get('height', np.nan), 'p2_height': loser_info.get('height', np.nan),
            'p1_hand': winner_info.get('hand', 'U'), 'p2_hand': loser_info.get('hand', 'U'),
            'h2h_p1_wins': h2h_hist['p1_wins'] if p1_id == h2h_key[0] else h2h_hist['p2_wins'],
            'h2h_p2_wins': h2h_hist['p2_wins'] if p1_id == h2h_key[0] else h2h_hist['p1_wins'],
            'p1_win_perc': winner_hist['wins'] / winner_hist['matches'] if winner_hist['matches'] > 0 else 0,
            'p2_win_perc': loser_hist['wins'] / loser_hist['matches'] if loser_hist['matches'] > 0 else 0,
            'p1_surface_win_perc': winner_hist['surface_wins'][surface] / winner_hist['surface_matches'][surface] if winner_hist['surface_matches'][surface] > 0 else 0,
            'p2_surface_win_perc': loser_hist['surface_wins'][surface] / loser_hist['surface_matches'][surface] if loser_hist['surface_matches'][surface] > 0 else 0,
            # --- NEW: Recent Form Features ---
            'p1_form_last_10': p1_form,
            'p2_form_last_10': p2_form,
        })
        
        # --- Update stats for the next match ---
        winner_hist['wins'] += 1
        winner_hist['matches'] += 1
        winner_hist['surface_wins'][surface] += 1
        winner_hist['surface_matches'][surface] += 1
        winner_hist['last_10_results'].append(1) # 1 for a win
        
        loser_hist['matches'] += 1
        loser_hist['surface_matches'][surface] += 1
        loser_hist['last_10_results'].append(0) # 0 for a loss

        h2h_hist['total'] += 1
        if winner_id == h2h_key[0]: h2h_hist['p1_wins'] += 1
        else: h2h_hist['p2_wins'] += 1
        h2h_stats[h2h_key] = h2h_hist

    return pd.DataFrame(features_list)

def main():
    parser = argparse.ArgumentParser(description="Build advanced historical player features with form and surface data.")
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--rankings_csv", required=True)
    parser.add_argument("--players_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    print("Loading data...")
    df_matches = pd.read_csv(args.matches_csv, low_memory=False)
    df_rankings = pd.read_csv(args.rankings_csv)
    df_players = pd.read_csv(args.players_csv, encoding='latin-1')
    
    df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'])
    df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'])
    df_matches['match_id'] = df_matches['tourney_id'] + '-' + df_matches['match_num'].astype(str)

    features_df = build_advanced_player_features(df_matches, df_rankings, df_players)
    
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving features to {args.output_csv}...")
    features_df.to_csv(args.output_csv, index=False)
    
    print(f"✅ Successfully created form-aware feature library at {args.output_csv}")

if __name__ == "__main__":
    main()