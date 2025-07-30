# src/tennis_betting_model/builders/player_mapper.py

import pandas as pd
import glob
import os
from pathlib import Path
from thefuzz import process
from tqdm import tqdm
import unidecode

from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    log_warning,
)


def get_initial_lastname(name):
    """Converts 'Novak Djokovic' to 'N Djokovic'."""
    parts = name.strip().split()
    if len(parts) == 2:
        return f"{parts[0][0]} {parts[-1]}"
    return name


def get_lastname(name):
    """Extracts the last name, e.g., 'Novak Djokovic' -> 'Djokovic'."""
    parts = name.strip().split()
    return parts[-1] if parts else name


def clean_name(name):
    """Removes accents, hyphens, and converts to lowercase."""
    if not isinstance(name, str):
        return ""
    return unidecode.unidecode(name).replace("-", " ").lower()


def _match_exact(unmatched_players, historical_players):
    """Pass 1: Exact Name Match."""
    exact_matches = pd.merge(
        unmatched_players.reset_index(),
        historical_players,
        left_on="runner_name",
        right_on="historical_name",
    )
    mappings = [
        {
            "betfair_id": row["runner_id"],
            "historical_id": row["historical_id"],
            "betfair_name": row["runner_name"],
            "matched_name": row["historical_name"],
            "confidence": 100,
            "method": "Exact",
        }
        for _, row in exact_matches.iterrows()
    ]
    unmatched_players.drop(
        index=exact_matches["runner_id"].tolist(), inplace=True, errors="ignore"
    )
    return mappings, unmatched_players


def _match_cleaned(unmatched_players, historical_players):
    """Pass 2: Exact Match on Cleaned Names."""
    unmatched_players["cleaned_name"] = unmatched_players["runner_name"].apply(
        clean_name
    )
    historical_players["cleaned_name"] = historical_players["historical_name"].apply(
        clean_name
    )
    cleaned_matches = pd.merge(
        unmatched_players.reset_index(), historical_players, on="cleaned_name"
    )
    cleaned_matches.drop_duplicates(subset=["historical_id"], keep=False, inplace=True)
    mappings = [
        {
            "betfair_id": row["runner_id"],
            "historical_id": row["historical_id"],
            "betfair_name": row["runner_name"],
            "matched_name": row["historical_name"],
            "confidence": 99.5,
            "method": "Exact-Cleaned",
        }
        for _, row in cleaned_matches.iterrows()
    ]
    unmatched_players.drop(
        index=cleaned_matches["runner_id"].tolist(), inplace=True, errors="ignore"
    )
    return mappings, unmatched_players


def _match_initial_lastname(unmatched_players, historical_players):
    """Pass 3: Initial + Last Name Match."""
    historical_players["initial_lastname"] = historical_players[
        "historical_name"
    ].apply(get_initial_lastname)
    unmatched_players["initial_lastname"] = unmatched_players["runner_name"].apply(
        get_initial_lastname
    )
    initial_matches = pd.merge(
        unmatched_players.reset_index(), historical_players, on="initial_lastname"
    )
    initial_matches.drop_duplicates(subset=["historical_id"], keep=False, inplace=True)
    mappings = [
        {
            "betfair_id": row["runner_id"],
            "historical_id": row["historical_id"],
            "betfair_name": row["runner_name"],
            "matched_name": row["historical_name"],
            "confidence": 99,
            "method": "Initial+Lastname",
        }
        for _, row in initial_matches.iterrows()
    ]
    unmatched_players.drop(
        index=initial_matches["runner_id"].tolist(), inplace=True, errors="ignore"
    )
    return mappings, unmatched_players


def _match_unique_lastname(unmatched_players, historical_players):
    """Pass 4: Unique Last Name Match."""
    historical_players["lastname"] = historical_players["historical_name"].apply(
        get_lastname
    )
    unmatched_players["lastname"] = unmatched_players["runner_name"].apply(get_lastname)
    lastname_counts = historical_players["lastname"].value_counts()
    unique_lastnames = lastname_counts[lastname_counts == 1].index.tolist()
    historical_unique_lastname = historical_players[
        historical_players["lastname"].isin(unique_lastnames)
    ]
    unique_lastname_matches = pd.merge(
        unmatched_players.reset_index(), historical_unique_lastname, on="lastname"
    )
    mappings = [
        {
            "betfair_id": row["runner_id"],
            "historical_id": row["historical_id"],
            "betfair_name": row["runner_name"],
            "matched_name": row["historical_name"],
            "confidence": 98,
            "method": "Unique Lastname",
        }
        for _, row in unique_lastname_matches.iterrows()
    ]
    unmatched_players.drop(
        index=unique_lastname_matches["runner_id"].tolist(),
        inplace=True,
        errors="ignore",
    )
    return mappings, unmatched_players


def _match_fuzzy(unmatched_players, historical_players, confidence_threshold, tour):
    """Pass 5: Fuzzy Match on the rest."""
    if unmatched_players.empty:
        return [], unmatched_players

    unique_historical = historical_players.drop_duplicates(
        subset=["historical_name"], keep="first"
    )
    historical_map = unique_historical.set_index("historical_name")["historical_id"]
    historical_list = historical_map.index.tolist()

    mappings = []
    for betfair_id, row in tqdm(
        unmatched_players.iterrows(),
        total=len(unmatched_players),
        desc=f"Fuzzy Matching ({tour.upper()})",
    ):
        betfair_name = row["runner_name"]
        if not isinstance(betfair_name, str) or not betfair_name.strip():
            continue

        best_match, score = process.extractOne(betfair_name, historical_list)
        if score >= confidence_threshold:
            mappings.append(
                {
                    "betfair_id": betfair_id,
                    "historical_id": historical_map[best_match],
                    "betfair_name": betfair_name,
                    "matched_name": best_match,
                    "confidence": score,
                    "method": "Fuzzy",
                }
            )
    return mappings, unmatched_players


def _run_mapping_passes(
    betfair_players: pd.DataFrame,
    historical_players: pd.DataFrame,
    confidence_threshold: int,
    tour: str,
) -> pd.DataFrame:
    """Runs the multi-pass matching logic."""
    unmatched = betfair_players.copy()
    all_mappings = []

    exact_mappings, unmatched = _match_exact(unmatched, historical_players)
    all_mappings.extend(exact_mappings)

    cleaned_mappings, unmatched = _match_cleaned(unmatched, historical_players)
    all_mappings.extend(cleaned_mappings)

    initial_mappings, unmatched = _match_initial_lastname(unmatched, historical_players)
    all_mappings.extend(initial_mappings)

    unique_mappings, unmatched = _match_unique_lastname(unmatched, historical_players)
    all_mappings.extend(unique_mappings)

    fuzzy_mappings, _ = _match_fuzzy(
        unmatched, historical_players, confidence_threshold, tour
    )
    all_mappings.extend(fuzzy_mappings)

    return pd.DataFrame(all_mappings)


def _load_historical_tour_data(raw_data_dir: Path, tour: str) -> pd.DataFrame:
    """Loads and consolidates historical match data for a specific tour (ATP or WTA)."""
    tour_files = glob.glob(
        os.path.join(raw_data_dir, f"tennis_{tour}", f"{tour}_matches_*.csv")
    )
    if not tour_files:
        log_warning(f"No historical match files found for tour: {tour}")
        return pd.DataFrame()

    df_historical_list = [pd.read_csv(f, low_memory=False) for f in tour_files]
    df_historical = pd.concat(df_historical_list, ignore_index=True)

    historical_winners = df_historical[["winner_id", "winner_name"]].rename(
        columns={"winner_id": "historical_id", "winner_name": "historical_name"}
    )
    historical_losers = df_historical[["loser_id", "loser_name"]].rename(
        columns={"loser_id": "historical_id", "loser_name": "historical_name"}
    )

    return pd.concat([historical_winners, historical_losers]).drop_duplicates().dropna()


def create_mapping_file(config: dict):
    log_info("--- Player Mapping File Generator (Historical Data Driven) ---")
    paths = config["data_paths"]
    mapping_params = config["mapping_params"]
    confidence_threshold = mapping_params.get("confidence_threshold", 85)

    log_info("Loading all unique players from RAW Betfair data...")
    betfair_odds_path = Path(paths["betfair_raw_odds"])
    if not betfair_odds_path.exists():
        log_error("Betfair RAW odds file not found. Please run 'prepare-data' first.")
        return

    df_betfair_odds = pd.read_csv(betfair_odds_path)
    betfair_unique_players = (
        df_betfair_odds[["selection_id", "selection_name"]]
        .drop_duplicates()
        .rename(columns={"selection_id": "runner_id", "selection_name": "runner_name"})
        .set_index("runner_id")
    )
    log_info(f"Found {len(betfair_unique_players)} unique players in Betfair data.")

    raw_data_dir = Path(paths["raw_data_dir"])
    atp_historical_players = _load_historical_tour_data(raw_data_dir, "atp")
    wta_historical_players = _load_historical_tour_data(raw_data_dir, "wta")

    log_info("\n--- Running Mapping vs. ATP historical data ---")
    atp_mappings = _run_mapping_passes(
        betfair_unique_players, atp_historical_players, confidence_threshold, tour="atp"
    )
    log_success(f"Found {len(atp_mappings)} potential ATP matches.")

    log_info("\n--- Running Mapping vs. WTA historical data ---")
    wta_mappings = _run_mapping_passes(
        betfair_unique_players, wta_historical_players, confidence_threshold, tour="wta"
    )
    log_success(f"Found {len(wta_mappings)} potential WTA matches.")

    final_mappings = pd.concat([atp_mappings, wta_mappings], ignore_index=True)
    final_mappings.sort_values(by="confidence", ascending=False, inplace=True)
    final_mappings.drop_duplicates(subset=["betfair_id"], keep="first", inplace=True)

    output_path = Path(paths["player_map"])
    final_mappings.to_csv(output_path, index=False)

    log_success(
        f"\nSuccessfully generated a combined mapping file with {len(final_mappings)} unique entries."
    )
    log_success(f"File saved to: {output_path}")
    log_warning(
        "IMPORTANT: Please manually review this file. It should now be highly accurate."
    )
