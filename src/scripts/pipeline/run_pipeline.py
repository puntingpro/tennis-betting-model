# src/scripts/pipeline/run_pipeline.py

import pandas as pd
import joblib
from decimal import Decimal, getcontext

getcontext().prec = 10

from src.scripts.utils.logger import setup_logging, log_info, log_warning
from src.scripts.utils.config import load_config
from src.scripts.utils.api import login_to_betfair, get_tennis_competitions, get_live_match_odds
from src.scripts.utils.common import get_most_recent_ranking

def process_markets(model, market_catalogues, market_book_lookup, player_info_lookup, df_rankings, betting_config):
    # ... (This function remains the same as the previous version)
    log_info("Building features, predicting, and detecting value...")
    value_bets = []
    ev_threshold = Decimal(str(betting_config['ev_threshold']))

    for market in market_catalogues:
        market_book = market_book_lookup.get(market.market_id)
        if not market_book or len(market.runners) != 2 or len(market_book.runners) != 2:
            continue

        try:
            p1_meta, p2_meta = market.runners[0], market.runners[1]
            p1_book, p2_book = market_book.runners[0], market_book.runners[1]

            p1_id, p2_id = p1_meta.selection_id, p2_meta.selection_id
            p1_info, p2_info = player_info_lookup.get(p1_id, {}), player_info_lookup.get(p2_id, {})

            match_date = pd.to_datetime(market.market_start_time).tz_convert('UTC')
            p1_rank = get_most_recent_ranking(df_rankings, p1_id, match_date)
            p2_rank = get_most_recent_ranking(df_rankings, p2_id, match_date)

            feature_dict = {
                'p1_rank': p1_rank, 'p2_rank': p2_rank, 'rank_diff': p1_rank - p2_rank if pd.notna(p1_rank) and pd.notna(p2_rank) else 0,
                'p1_height': p1_info.get('height', 0), 'p2_height': p2_info.get('height', 0),
                'h2h_p1_wins': 0, 'h2h_p2_wins': 0, 'h2h_win_perc_p1': 0, 'p1_win_perc': 0, 'p2_win_perc': 0,
                'p1_surface_win_perc': 0, 'p2_surface_win_perc': 0,
                'p1_hand_L': 1 if p1_info.get('hand') == 'L' else 0, 'p1_hand_R': 1 if p1_info.get('hand') == 'R' else 0, 'p1_hand_U': 1 if p1_info.get('hand') == 'U' else 0,
                'p2_hand_L': 1 if p2_info.get('hand') == 'L' else 0, 'p2_hand_R': 1 if p2_info.get('hand') == 'R' else 0, 'p2_hand_U': 1 if p2_info.get('hand') == 'U' else 0,
            }
            features = pd.DataFrame([feature_dict], columns=model.feature_names_in_).fillna(0)
            
            win_prob_p1 = Decimal(str(model.predict_proba(features)[0][1]))
            win_prob_p2 = Decimal('1.0') - win_prob_p1

            if p1_book.ex.available_to_back:
                p1_odds = Decimal(str(p1_book.ex.available_to_back[0].price))
                p1_ev = (win_prob_p1 * p1_odds) - Decimal('1.0')
                if p1_ev > ev_threshold:
                    value_bets.append({'match': f"{market.competition.name} - {market.event.name}", 'player_name': p1_meta.runner_name, 'odds': float(p1_odds), 'Model Prob': f"{win_prob_p1:.0%}", 'EV': f"{p1_ev:+.2%}"})
            
            if p2_book.ex.available_to_back:
                p2_odds = Decimal(str(p2_book.ex.available_to_back[0].price))
                p2_ev = (win_prob_p2 * p2_odds) - Decimal('1.0')
                if p2_ev > ev_threshold:
                     value_bets.append({'match': f"{market.competition.name} - {market.event.name}", 'player_name': p2_meta.runner_name, 'odds': float(p2_odds), 'Model Prob': f"{win_prob_p2:.0%}", 'EV': f"{p2_ev:+.2%}"})
        except Exception as e:
            log_info(f"Skipping market due to processing error: {e}")
            continue
    
    return value_bets


def main(args):
    """
    Main pipeline function: Connects to API, loads data, and calls the processing logic.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']
    betting_config = config['betting']

    # --- UPDATED: DRY-RUN LOGIC ---
    if args.dry_run:
        log_warning("🚀 Running in DRY-RUN mode. No real bets will be placed.")
    else:
        log_info("🚀 Running in LIVE mode. Real bets will be placed.")

    log_info(f"Targeting profitable tournaments containing: {betting_config['profitable_tournaments']}")

    log_info("Loading model and all required data sources...")
    model = joblib.load(paths['model'])
    
    df_atp_players = pd.read_csv("data/tennis_atp/atp_players.csv", encoding='latin-1')
    df_wta_players = pd.read_csv("data/tennis_wta/wta_players.csv", encoding='latin-1')
    df_players = pd.concat([df_atp_players, df_wta_players], ignore_index=True)
    df_players = df_players.drop_duplicates(subset=['player_id'], keep='first')
    player_info_lookup = df_players.set_index('player_id').to_dict('index')

    df_atp_rankings = pd.read_csv("data/processed/atp_rankings_consolidated.csv")
    df_wta_rankings = pd.read_csv("data/processed/wta_rankings_consolidated.csv")
    df_rankings = pd.concat([df_atp_rankings, df_wta_rankings], ignore_index=True)
    
    df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'], utc=True)
    df_rankings = df_rankings.sort_values(by='ranking_date')
    
    log_info("✅ All data loaded successfully.")

    trading = login_to_betfair(config)
    try:
        target_competition_ids = get_tennis_competitions(trading, betting_config['profitable_tournaments'])
        if not target_competition_ids:
            log_info("No live competitions found. Exiting.")
            return
            
        log_info(f"Found {len(target_competition_ids)} target competitions to scan.")
        market_catalogues, market_book_lookup = get_live_match_odds(trading, target_competition_ids)
        if not market_catalogues:
            log_info("No live matches found in profitable tournament types. Exiting.")
            return

        value_bets = process_markets(model, market_catalogues, market_book_lookup, player_info_lookup, df_rankings, betting_config)

        if value_bets:
            log_info("✅ Value Bets Found ✅")
            print(pd.DataFrame(value_bets).to_string(index=False))
            # --- UPDATED: DRY-RUN LOGIC ---
            if not args.dry_run:
                log_warning("Placing live bets...")
                # In a real application, the code to place bets would go here.
        else:
            log_info("--- No Value Bets Found in Targeted Competitions ---")
    finally:
        trading.logout()
        log_info("\nLogged out.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    # --- ADDED FOR STANDALONE EXECUTION ---
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without placing bets.")
    args = parser.parse_args()
    main(args)