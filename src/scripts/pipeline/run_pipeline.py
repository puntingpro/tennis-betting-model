# src/scripts/pipeline/run_pipeline.py

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import betfairlightweight
from unidecode import unidecode

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.betting_math import add_ev_and_kelly
from src.scripts.utils.common import get_most_recent_ranking
from src.scripts.utils.logger import setup_logging, log_info
from src.scripts.utils.config import load_config
from src.scripts.utils.api import login_to_betfair, get_tennis_competitions, get_live_match_odds

def main(args):
    """
    Main pipeline function: Uses the best model to find value bets in high-ROI tournaments.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']
    betting_config = config['betting']

    log_info(f"Targeting profitable tournaments containing: {betting_config['profitable_tournaments']}")

    # --- Load Final Model & All Data ---
    log_info("Loading model and all required data sources...")
    model = joblib.load(paths['model'])
    
    # This data loading assumes raw data is available.
    # In a production system, these might also be consolidated first.
    df_atp_players = pd.read_csv("data/tennis_atp/atp_players.csv", encoding='latin-1')
    df_wta_players = pd.read_csv("data/tennis_wta/wta_players.csv", encoding='latin-1')
    df_players = pd.concat([df_atp_players, df_wta_players], ignore_index=True)
    df_players = df_players.drop_duplicates(subset=['player_id'], keep='first')
    player_info_lookup = df_players.set_index('player_id').to_dict('index')

    df_atp_rankings = pd.read_csv("data/processed/atp_rankings_consolidated.csv")
    df_wta_rankings = pd.read_csv("data/processed/wta_rankings_consolidated.csv")
    df_rankings = pd.concat([df_atp_rankings, df_wta_rankings], ignore_index=True)
    df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'])
    df_rankings = df_rankings.sort_values(by='ranking_date')
    
    log_info("✅ All data loaded successfully.")

    # --- Login & Find Target Competitions ---
    trading = login_to_betfair(config)

    target_competition_ids = get_tennis_competitions(trading, betting_config['profitable_tournaments'])
    
    if not target_competition_ids:
        log_info("No live matches found in profitable tournament types. Exiting.")
        trading.logout()
        return
    log_info(f"Found {len(target_competition_ids)} target competitions to scan.")

    # --- Fetch Matches, Odds, and Find Value ---
    market_catalogues, market_book_lookup = get_live_match_odds(trading, target_competition_ids)
    if not market_catalogues:
        log_info("No live matches found in profitable tournament types. Exiting.")
        trading.logout()
        return

    log_info("Building features, predicting, and detecting value...")
    value_bets = []
    for market in market_catalogues:
        market_book = market_book_lookup.get(market.market_id)
        if not market_book or len(market.runners) != 2 or len(market_book.runners) != 2: continue

        try:
            surface = market.market_name # Extract surface from market name
            
            p1_meta, p2_meta = market.runners[0], market.runners[1]
            p1_book, p2_book = market_book.runners[0], market_book.runners[1]
            
            p1_id, p2_id = p1_meta.selection_id, p2_meta.selection_id
            p1_info, p2_info = player_info_lookup.get(p1_id, {}), player_info_lookup.get(p2_id, {})
            
            match_date = pd.to_datetime(market.market_start_time)
            p1_rank = get_most_recent_ranking(df_rankings, p1_id, match_date)
            p2_rank = get_most_recent_ranking(df_rankings, p2_id, match_date)

            feature_dict = {
                'p1_rank': p1_rank, 'p2_rank': p2_rank, 'rank_diff': p1_rank - p2_rank if pd.notna(p1_rank) and pd.notna(p2_rank) else 0,
                'p1_height': p1_info.get('height', 0), 'p2_height': p2_info.get('height', 0),
                'h2h_p1_wins': 0, 'h2h_p2_wins': 0, 'h2h_win_perc_p1': 0, 'p1_win_perc': 0, 'p2_win_perc': 0,
                'p1_surface_win_perc': 0, 'p2_surface_win_perc': 0, # Placeholder for live data
                'p1_hand_L': 1 if p1_info.get('hand') == 'L' else 0, 'p1_hand_R': 1 if p1_info.get('hand') == 'R' else 0, 'p1_hand_U': 1 if p1_info.get('hand') == 'U' else 0,
                'p2_hand_L': 1 if p2_info.get('hand') == 'L' else 0, 'p2_hand_R': 1 if p2_info.get('hand') == 'R' else 0, 'p2_hand_U': 1 if p2_info.get('hand') == 'U' else 0,
            }
            features = pd.DataFrame([feature_dict], columns=model.feature_names_in_).fillna(0)
            
            win_prob_p1 = model.predict_proba(features)[0][1]

            if p1_book.ex.available_to_back:
                p1_odds = p1_book.ex.available_to_back[0].price
                p1_ev = (win_prob_p1 * p1_odds) - 1
                if p1_ev > betting_config['ev_threshold']:
                    value_bets.append({'match': f"{market.competition.name} - {market.event.name}", 'player_name': p1_meta.runner_name, 'odds': p1_odds, 'Model Prob': f"{win_prob_p1:.0%}", 'EV': f"{p1_ev:+.2%}"})
            
            if p2_book.ex.available_to_back:
                p2_odds = p2_book.ex.available_to_back[0].price
                p2_ev = ((1 - win_prob_p1) * p2_odds) - 1
                if p2_ev > betting_config['ev_threshold']:
                     value_bets.append({'match': f"{market.competition.name} - {market.event.name}", 'player_name': p2_meta.runner_name, 'odds': p2_odds, 'Model Prob': f"{1 - win_prob_p1:.0%}", 'EV': f"{p2_ev:+.2%}"})
        except Exception as e:
            continue

    if value_bets:
        print("\n--- ✅ Value Bets Found ✅ ---")
        value_bets_df = pd.DataFrame(value_bets)
        print(value_bets_df.to_string(index=False))
    else:
        print("\n--- No Value Bets Found in Targeted Competitions ---")
        
    trading.logout()
    log_info("\nLogged out.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)