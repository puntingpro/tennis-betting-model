# src/tennis_betting_model/builders/build_player_features.py

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Any
from collections import defaultdict

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
)
from tennis_betting_model.utils.common import get_surface, get_most_recent_ranking
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.utils.constants import ELO_INITIAL_RATING


def _load_data(
    paths: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all necessary dataframes for feature building."""
    log_info("Loading consolidated data and other required files...")
    try:
        df_matches_raw = pd.read_csv(paths["betfair_match_log"])
        df_rankings = pd.read_csv(paths["consolidated_rankings"])
        df_players = pd.read_csv(paths["raw_players"], encoding="latin-1")
        df_elo = pd.read_csv(paths["elo_ratings"])

        df_players["player_id"] = pd.to_numeric(
            df_players["player_id"], errors="coerce"
        )
        df_players.dropna(subset=["player_id"], inplace=True)
        df_players["player_id"] = df_players["player_id"].astype("int64")

        df_rankings["ranking_date"] = pd.to_datetime(
            df_rankings["ranking_date"], utc=True
        )
        df_rankings = df_rankings.sort_values("ranking_date")

        df_matches_raw["tourney_date"] = pd.to_datetime(
            df_matches_raw["tourney_date"], utc=True
        )

        df_matches_raw["winner_historical_id"] = pd.to_numeric(
            df_matches_raw["winner_historical_id"], errors="coerce"
        )
        df_matches_raw["loser_historical_id"] = pd.to_numeric(
            df_matches_raw["loser_historical_id"], errors="coerce"
        )
        df_matches_raw.dropna(
            subset=["winner_historical_id", "loser_historical_id"], inplace=True
        )
        df_matches_raw = df_matches_raw.astype(
            {"winner_historical_id": "int64", "loser_historical_id": "int64"}
        )
        df_matches_raw["match_id"] = df_matches_raw["match_id"].astype(str)
        df_elo["match_id"] = df_elo["match_id"].astype(str)

        df_matches_raw["surface"] = df_matches_raw["tourney_name"].apply(get_surface)

        return df_matches_raw, df_rankings, df_players, df_elo
    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
        raise


def main(args):
    """Main workflow for building all player features using a stateful, chronological approach."""
    setup_logging()

    config = load_config(args.config)
    paths = config["data_paths"]

    df_matches, df_rankings, df_players, df_elo = _load_data(paths)

    df_matches = validate_data(
        df_matches, "betfair_match_log", "Initial Betfair Match Log"
    )

    log_info("--- Starting High-Performance Feature Engineering ---")

    player_info_lookup = (
        df_players[["player_id", "hand"]]
        .drop_duplicates("player_id")
        .set_index("player_id")
        .to_dict("index")
    )

    df_elo_lookup = df_elo.set_index("match_id")

    # Sort matches chronologically to process them in order
    df_matches = df_matches.sort_values("tourney_date").reset_index(drop=True)

    # Initialize state dictionaries
    player_stats: Dict[int, Any] = defaultdict(
        lambda: {
            "matches_played": 0,
            "wins": 0,
            "surface_matches": defaultdict(int),
            "surface_wins": defaultdict(int),
            "recent_matches": [],
        }
    )
    h2h_stats: Dict[str, Any] = defaultdict(lambda: {"p1_wins": 0, "p2_wins": 0})
    all_features = []

    for row in tqdm(
        df_matches.itertuples(),
        total=len(df_matches),
        desc="Building Historical Features (Chronologically)",
    ):
        winner_id = row.winner_historical_id
        loser_id = row.loser_historical_id
        match_date = row.tourney_date
        surface = row.surface
        match_id = row.match_id

        p1_id = min(winner_id, loser_id)
        p2_id = max(winner_id, loser_id)

        if p1_id == p2_id:
            continue

        # --- Get PRE-MATCH stats from the state dictionaries ---
        p1_stats = player_stats[p1_id]
        p2_stats = player_stats[p2_id]
        h2h_key = f"{p1_id}-{p2_id}"
        h2h = h2h_stats[h2h_key]

        # Rankings and Elo
        p1_rank = get_most_recent_ranking(df_rankings, p1_id, match_date)
        p2_rank = get_most_recent_ranking(df_rankings, p2_id, match_date)
        try:
            match_elo = df_elo_lookup.loc[match_id]
            p1_elo = match_elo["p1_elo"]
            p2_elo = match_elo["p2_elo"]
        except KeyError:
            p1_elo = ELO_INITIAL_RATING
            p2_elo = ELO_INITIAL_RATING

        # Win percentages
        p1_win_perc = (
            p1_stats["wins"] / p1_stats["matches_played"]
            if p1_stats["matches_played"] > 0
            else 0
        )
        p2_win_perc = (
            p2_stats["wins"] / p2_stats["matches_played"]
            if p2_stats["matches_played"] > 0
            else 0
        )
        p1_surface_win_perc = (
            p1_stats["surface_wins"][surface] / p1_stats["surface_matches"][surface]
            if p1_stats["surface_matches"][surface] > 0
            else 0
        )
        p2_surface_win_perc = (
            p2_stats["surface_wins"][surface] / p2_stats["surface_matches"][surface]
            if p2_stats["surface_matches"][surface] > 0
            else 0
        )

        # Form and fatigue
        p1_recent_matches = [
            d for d in p1_stats["recent_matches"] if (match_date - d[0]).days < 30
        ]
        p2_recent_matches = [
            d for d in p2_stats["recent_matches"] if (match_date - d[0]).days < 30
        ]
        p1_form = (
            sum(1 for date, won in p1_recent_matches[-10:] if won)
            / len(p1_recent_matches[-10:])
            if p1_recent_matches
            else 0
        )
        p2_form = (
            sum(1 for date, won in p2_recent_matches[-10:] if won)
            / len(p2_recent_matches[-10:])
            if p2_recent_matches
            else 0
        )

        # FIX: Use the unpacked date 'd' directly, not d[0], as the loop already unpacks the tuple
        p1_matches_last_7 = sum(
            1 for d, _ in p1_recent_matches if (match_date - d).days <= 7
        )
        p2_matches_last_7 = sum(
            1 for d, _ in p2_recent_matches if (match_date - d).days <= 7
        )
        p1_matches_last_14 = sum(
            1 for d, _ in p1_recent_matches if (match_date - d).days <= 14
        )
        p2_matches_last_14 = sum(
            1 for d, _ in p2_recent_matches if (match_date - d).days <= 14
        )

        # Assemble feature dictionary
        feature_dict = {
            "market_id": match_id,
            "tourney_date": match_date,
            "tourney_name": row.tourney_name,
            "surface": surface,
            "p1_id": p1_id,
            "p2_id": p2_id,
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "rank_diff": p1_rank - p2_rank,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,
            "elo_diff": p1_elo - p2_elo,
            "p1_win_perc": p1_win_perc,
            "p2_win_perc": p2_win_perc,
            "p1_surface_win_perc": p1_surface_win_perc,
            "p2_surface_win_perc": p2_surface_win_perc,
            "p1_form": p1_form,
            "p2_form": p2_form,
            "p1_matches_last_7_days": p1_matches_last_7,
            "p2_matches_last_7_days": p2_matches_last_7,
            "p1_matches_last_14_days": p1_matches_last_14,
            "p2_matches_last_14_days": p2_matches_last_14,
            "fatigue_diff_7_days": p1_matches_last_7 - p2_matches_last_7,
            "fatigue_diff_14_days": p1_matches_last_14 - p2_matches_last_14,
            "h2h_p1_wins": h2h["p1_wins"],
            "h2h_p2_wins": h2h["p2_wins"],
            "p1_hand": player_info_lookup.get(p1_id, {}).get("hand", "U"),
            "p2_hand": player_info_lookup.get(p2_id, {}).get("hand", "U"),
            "winner": 1 if p1_id == winner_id else 0,
        }
        all_features.append(feature_dict)

        # --- UPDATE STATE for the next iteration ---
        # Winner stats
        player_stats[winner_id]["matches_played"] += 1
        player_stats[winner_id]["wins"] += 1
        player_stats[winner_id]["surface_matches"][surface] += 1
        player_stats[winner_id]["surface_wins"][surface] += 1
        player_stats[winner_id]["recent_matches"].append((match_date, True))

        # Loser stats
        player_stats[loser_id]["matches_played"] += 1
        player_stats[loser_id]["surface_matches"][surface] += 1
        player_stats[loser_id]["recent_matches"].append((match_date, False))

        # H2H stats
        if p1_id == winner_id:
            h2h_stats[h2h_key]["p1_wins"] += 1
        else:
            h2h_stats[h2h_key]["p2_wins"] += 1

    final_df = pd.DataFrame(all_features)

    if final_df.empty:
        log_error("No features were generated. The resulting DataFrame is empty.")
        return

    elo_cols = ["p1_elo", "p2_elo", "elo_diff"]
    for col in elo_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
        final_df[col] = final_df[col].fillna(ELO_INITIAL_RATING)

    validated_features = validate_data(final_df, "final_features", "Final Feature Set")

    output_path = Path(paths["consolidated_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL features to {output_path}...")
    validated_features.to_csv(output_path, index=False)
    log_success(f"✅ Successfully created FINAL feature library at {output_path}")
