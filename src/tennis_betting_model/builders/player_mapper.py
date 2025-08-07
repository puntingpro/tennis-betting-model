# src/tennis_betting_model/builders/player_mapper.py

import pandas as pd
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
from tennis_betting_model.utils.schema import validate_data
from tennis_betting_model.utils.data_loader import DataLoader
from tennis_betting_model.utils.config_schema import MappingParams, DataPaths


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


class PlayerMapper:
    def __init__(self, betfair_players, historical_players, confidence_threshold):
        self.unmatched = betfair_players.copy()
        self.historical = historical_players
        self.confidence_threshold = confidence_threshold
        self.mappings = []

    def run(self, tour: str):
        self._match_exact()
        self._match_cleaned()
        self._match_initial_lastname()
        self._match_unique_lastname()
        self._match_fuzzy(tour)
        return pd.DataFrame(self.mappings)

    def _match_exact(self):
        """Pass 1: Exact Name Match."""
        exact_matches = pd.merge(
            self.unmatched.reset_index(),
            self.historical,
            left_on="runner_name",
            right_on="historical_name",
        )
        new_mappings = [
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
        self.mappings.extend(new_mappings)
        self.unmatched.drop(
            index=exact_matches["runner_id"].tolist(), inplace=True, errors="ignore"
        )

    def _match_cleaned(self):
        """Pass 2: Exact Match on Cleaned Names."""
        self.unmatched["cleaned_name"] = self.unmatched["runner_name"].apply(clean_name)
        self.historical["cleaned_name"] = self.historical["historical_name"].apply(
            clean_name
        )
        cleaned_matches = pd.merge(
            self.unmatched.reset_index(), self.historical, on="cleaned_name"
        )
        cleaned_matches.drop_duplicates(
            subset=["historical_id"], keep=False, inplace=True
        )
        new_mappings = [
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
        self.mappings.extend(new_mappings)
        self.unmatched.drop(
            index=cleaned_matches["runner_id"].tolist(), inplace=True, errors="ignore"
        )

    def _match_initial_lastname(self):
        """Pass 3: Initial + Last Name Match."""
        self.historical["initial_lastname"] = self.historical["historical_name"].apply(
            get_initial_lastname
        )
        self.unmatched["initial_lastname"] = self.unmatched["runner_name"].apply(
            get_initial_lastname
        )
        initial_matches = pd.merge(
            self.unmatched.reset_index(), self.historical, on="initial_lastname"
        )
        initial_matches.drop_duplicates(
            subset=["historical_id"], keep=False, inplace=True
        )
        new_mappings = [
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
        self.mappings.extend(new_mappings)
        self.unmatched.drop(
            index=initial_matches["runner_id"].tolist(), inplace=True, errors="ignore"
        )

    def _match_unique_lastname(self):
        """Pass 4: Unique Last Name Match."""
        self.historical["lastname"] = self.historical["historical_name"].apply(
            get_lastname
        )
        self.unmatched["lastname"] = self.unmatched["runner_name"].apply(get_lastname)
        lastname_counts = self.historical["lastname"].value_counts()
        unique_lastnames = lastname_counts[lastname_counts == 1].index.tolist()
        historical_unique_lastname = self.historical[
            self.historical["lastname"].isin(unique_lastnames)
        ]
        unique_lastname_matches = pd.merge(
            self.unmatched.reset_index(), historical_unique_lastname, on="lastname"
        )
        new_mappings = [
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
        self.mappings.extend(new_mappings)
        self.unmatched.drop(
            index=unique_lastname_matches["runner_id"].tolist(),
            inplace=True,
            errors="ignore",
        )

    def _match_fuzzy(self, tour: str):
        """Pass 5: Fuzzy Match on the rest."""
        if self.unmatched.empty:
            return

        unique_historical = self.historical.drop_duplicates(
            subset=["historical_name"], keep="first"
        )
        historical_map = unique_historical.set_index("historical_name")["historical_id"]
        historical_list = historical_map.index.tolist()

        new_mappings = []
        for betfair_id, row in tqdm(
            self.unmatched.iterrows(),
            total=len(self.unmatched),
            desc=f"Fuzzy Matching ({tour.upper()})",
        ):
            betfair_name = row["runner_name"]
            if not isinstance(betfair_name, str) or not betfair_name.strip():
                continue

            best_match, score = process.extractOne(betfair_name, historical_list)
            if score >= self.confidence_threshold:
                new_mappings.append(
                    {
                        "betfair_id": betfair_id,
                        "historical_id": historical_map[best_match],
                        "betfair_name": betfair_name,
                        "matched_name": best_match,
                        "confidence": score,
                        "method": "Fuzzy",
                    }
                )
        self.mappings.extend(new_mappings)


def run_create_mapping_file(data_paths: DataPaths, mapping_params: MappingParams):
    log_info("--- Player Mapping File Generator (Historical Data Driven) ---")
    confidence_threshold = mapping_params.confidence_threshold
    data_loader = DataLoader(data_paths)

    log_info("Loading all unique players from RAW Betfair data...")
    betfair_odds_path = Path(data_paths.betfair_raw_odds)
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

    historical_players = data_loader.load_historical_player_data()

    log_info("\n--- Running Mapping vs. All historical data ---")
    mapper = PlayerMapper(
        betfair_unique_players, historical_players, confidence_threshold
    )
    all_mappings = mapper.run("All")
    log_success(f"Found {len(all_mappings)} potential matches.")

    final_mappings = all_mappings.sort_values(
        by="confidence", ascending=False
    ).drop_duplicates(subset=["betfair_id"], keep="first")

    output_path = Path(data_paths.player_map)
    final_mappings.to_csv(output_path, index=False)

    log_success(
        f"\nSuccessfully generated a combined mapping file with {len(final_mappings)} unique entries."
    )
    log_success(f"File saved to: {output_path}")
    log_warning(
        "IMPORTANT: Please manually review this file. It should now be highly accurate."
    )

    # Validate the final dataframe
    validate_data(final_mappings, "player_map", "Final Player Map")
