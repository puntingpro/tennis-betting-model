import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict, deque

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.data_loader import load_all_pipeline_data
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    setup_logging,
    log_error,
    log_warning,
)
from tennis_betting_model.utils.common import get_surface, get_most_recent_ranking
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.utils.constants import ELO_INITIAL_RATING


def _generate_features_chronologically(
    df_matches: pd.DataFrame,
    df_rankings: pd.DataFrame,
    df_elo_lookup: pd.DataFrame,
    player_info_lookup: dict,
) -> pd.DataFrame:
    """
    Processes matches chronologically to build point-in-time features.
    """
    player_stats: Dict[int, Any] = defaultdict(
        lambda: {
            "matches_played": 0,
            "wins": 0,
            "surface_matches": defaultdict(int),
            "surface_wins": defaultdict(int),
            "recent_matches": [],
            "elo_history": deque(maxlen=5),
        }
    )
    h2h_stats: Dict[str, Any] = defaultdict(lambda: {"p1_wins": 0, "p2_wins": 0})
    all_features = []

    df_matches["surface"] = df_matches["tourney_name"].apply(get_surface)

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

        p1_stats = player_stats[p1_id]
        p2_stats = player_stats[p2_id]
        h2h_key = f"{p1_id}-{p2_id}"
        h2h = h2h_stats[h2h_key]

        # --- FIX START: Make Elo lookup robust to duplicates ---
        try:
            match_elo_data = df_elo_lookup.loc[match_id]
            # If duplicates exist, .loc returns a DataFrame; otherwise, a Series
            if isinstance(match_elo_data, pd.DataFrame):
                # If we find duplicates, log it and use the first entry
                log_warning(
                    f"Duplicate match_id '{match_id}' found in Elo data. Using first entry."
                )
                match_elo = match_elo_data.iloc[0]
            else:
                match_elo = match_elo_data

            p1_elo = match_elo["p1_elo"]
            p2_elo = match_elo["p2_elo"]
        except KeyError:
            p1_elo = ELO_INITIAL_RATING
            p2_elo = ELO_INITIAL_RATING
        # --- FIX END ---

        p1_elo_hist = list(p1_stats["elo_history"])
        p1_elo_momentum = (
            p1_elo - (sum(p1_elo_hist) / len(p1_elo_hist)) if p1_elo_hist else 0
        )
        p2_elo_hist = list(p2_stats["elo_history"])
        p2_elo_momentum = (
            p2_elo - (sum(p2_elo_hist) / len(p2_elo_hist)) if p2_elo_hist else 0
        )

        # Other features...
        p1_rank = get_most_recent_ranking(df_rankings, p1_id, match_date)
        p2_rank = get_most_recent_ranking(df_rankings, p2_id, match_date)
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
            "p1_elo_momentum": p1_elo_momentum,
            "p2_elo_momentum": p2_elo_momentum,
            "elo_momentum_diff": p1_elo_momentum - p2_elo_momentum,
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

        # UPDATE STATE
        player_stats[winner_id]["matches_played"] += 1
        player_stats[winner_id]["wins"] += 1
        player_stats[winner_id]["surface_matches"][surface] += 1
        player_stats[winner_id]["surface_wins"][surface] += 1
        player_stats[winner_id]["recent_matches"].append((match_date, True))
        player_stats[winner_id]["elo_history"].append(
            p1_elo if winner_id == p1_id else p2_elo
        )

        player_stats[loser_id]["matches_played"] += 1
        player_stats[loser_id]["surface_matches"][surface] += 1
        player_stats[loser_id]["recent_matches"].append((match_date, False))
        player_stats[loser_id]["elo_history"].append(
            p1_elo if loser_id == p1_id else p2_elo
        )

        if p1_id == winner_id:
            h2h_stats[h2h_key]["p1_wins"] += 1
        else:
            h2h_stats[h2h_key]["p2_wins"] += 1

    return pd.DataFrame(all_features)


def main(args):
    """Main workflow for building all player features using a stateful, chronological approach."""
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]

    try:
        df_matches, df_rankings, _, df_elo, player_info_lookup = load_all_pipeline_data(
            paths
        )
    except FileNotFoundError:
        return

    log_info("--- Starting High-Performance Feature Engineering ---")
    df_elo_lookup = df_elo.set_index("match_id")
    df_matches = df_matches.sort_values("tourney_date").reset_index(drop=True)

    final_df = _generate_features_chronologically(
        df_matches, df_rankings, df_elo_lookup, player_info_lookup
    )

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
