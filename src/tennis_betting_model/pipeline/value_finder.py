# src/scripts/pipeline/value_finder.py

import pandas as pd
from decimal import Decimal
from scripts.utils.logger import log_info, log_success
from scripts.utils.common import get_most_recent_ranking
from scripts.pipeline.feature_engineering import (
    get_h2h_stats,
    get_player_form_and_win_perc,
)
from scripts.utils.alerter import send_alert


def process_markets(
    model,
    market_catalogues,
    market_book_lookup,
    player_info_lookup,
    df_rankings,
    df_matches,
    betting_config,
):
    """
    Builds features for live markets, makes predictions, and identifies value bets.
    """
    log_info(f"Processing {len(market_catalogues)} live markets...")
    value_bets = []
    ev_threshold = Decimal(str(betting_config["ev_threshold"]))

    for market in market_catalogues:
        market_id = market.market_id
        market_book = market_book_lookup.get(market_id)
        surface = market.market_name.split(" ")[-1]

        if not market_book or len(market.runners) != 2 or len(market_book.runners) != 2:
            log_info(f"Skipping market {market_id}: Invalid number of runners.")
            continue

        try:
            p1_meta, p2_meta = market.runners[0], market.runners[1]
            p1_book, p2_book = market_book.runners[0], market_book.runners[1]

            p1_id, p2_id = int(p1_meta.selection_id), int(p2_meta.selection_id)
            p1_info, p2_info = player_info_lookup.get(
                p1_id, {}
            ), player_info_lookup.get(p2_id, {})

            match_date = pd.to_datetime(market.market_start_time).tz_convert("UTC")
            p1_rank = get_most_recent_ranking(df_rankings, p1_id, match_date)
            p2_rank = get_most_recent_ranking(df_rankings, p2_id, match_date)

            p1_win_perc, p1_surface_win_perc, _ = get_player_form_and_win_perc(
                df_matches, p1_id, surface, match_date
            )
            p2_win_perc, p2_surface_win_perc, _ = get_player_form_and_win_perc(
                df_matches, p2_id, surface, match_date
            )
            h2h_p1_wins, h2h_p2_wins = get_h2h_stats(
                df_matches, p1_id, p2_id, match_date
            )
            h2h_total = h2h_p1_wins + h2h_p2_wins
            h2h_win_perc_p1 = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5

            feature_dict = {
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "rank_diff": p1_rank - p2_rank
                if pd.notna(p1_rank) and pd.notna(p2_rank)
                else 0,
                "p1_height": p1_info.get("height", 0),
                "p2_height": p2_info.get("height", 0),
                "h2h_p1_wins": h2h_p1_wins,
                "h2h_p2_wins": h2h_p2_wins,
                "h2h_win_perc_p1": h2h_win_perc_p1,
                "p1_win_perc": p1_win_perc,
                "p2_win_perc": p2_win_perc,
                "p1_surface_win_perc": p1_surface_win_perc,
                "p2_surface_win_perc": p2_surface_win_perc,
                "p1_hand_L": 1 if p1_info.get("hand") == "L" else 0,
                "p1_hand_R": 1 if p1_info.get("hand") == "R" else 0,
                "p1_hand_U": 1 if p1_info.get("hand") == "U" else 0,
                "p2_hand_L": 1 if p2_info.get("hand") == "L" else 0,
                "p2_hand_R": 1 if p2_info.get("hand") == "R" else 0,
                "p2_hand_U": 1 if p2_info.get("hand") == "U" else 0,
            }
            features = pd.DataFrame(
                [feature_dict], columns=model.feature_names_in_
            ).fillna(0)

            win_prob_p1 = Decimal(str(model.predict_proba(features)[0][1]))
            win_prob_p2 = Decimal("1.0") - win_prob_p1

            if p1_book.ex.available_to_back:
                p1_odds = Decimal(str(p1_book.ex.available_to_back[0].price))
                p1_ev = (win_prob_p1 * p1_odds) - Decimal("1.0")
                if p1_ev > ev_threshold:
                    bet_info = {
                        "match": f"{market.competition.name} - {market.event.name}",
                        "player_name": p1_meta.runner_name,
                        "odds": float(p1_odds),
                        "Model Prob": f"{win_prob_p1:.2%}",
                        "EV": f"{p1_ev:+.2%}",
                    }
                    log_success(
                        f"VALUE BET FOUND: {bet_info['player_name']} @ {bet_info['odds']} (EV: {bet_info['EV']})"
                    )
                    value_bets.append(bet_info)

            if p2_book.ex.available_to_back:
                p2_odds = Decimal(str(p2_book.ex.available_to_back[0].price))
                p2_ev = (win_prob_p2 * p2_odds) - Decimal("1.0")
                if p2_ev > ev_threshold:
                    bet_info = {
                        "match": f"{market.competition.name} - {market.event.name}",
                        "player_name": p2_meta.runner_name,
                        "odds": float(p2_odds),
                        "Model Prob": f"{win_prob_p2:.2%}",
                        "EV": f"{p2_ev:+.2%}",
                    }
                    log_success(
                        f"VALUE BET FOUND: {bet_info['player_name']} @ {bet_info['odds']} (EV: {bet_info['EV']})"
                    )
                    value_bets.append(bet_info)
        except Exception as e:
            log_info(f"Skipping market {market_id} due to processing error: {e}")
            continue

    # --- ADDED: Send an alert if value bets are found ---
    if value_bets:
        bet_df = pd.DataFrame(value_bets)
        send_alert(bet_df.to_string(index=False))

    return value_bets
