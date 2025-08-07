# src/tennis_betting_model/builders/build_elo_ratings.py

from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any
from collections import defaultdict  # Import defaultdict

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.common import get_surface
from tennis_betting_model.utils.logger import log_warning, log_info


@dataclass
class EloCalculator:
    k_factor: int
    rating_diff_factor: int
    initial_rating: int = 1500
    # --- FIX START: Use defaultdict to handle any surface type, including 'Unknown' ---
    elo_ratings: dict[str, dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # --- FIX END ---

    def get_player_rating(self, player_id: int, surface: str) -> float:
        """Gets a player's rating for a specific surface. Creates the surface entry if it doesn't exist."""
        return self.elo_ratings[surface].get(player_id, self.initial_rating)

    def update_ratings(self, winner_id: int, loser_id: int, surface: str) -> None:
        """Updates player ratings for a specific surface."""
        winner_rating = self.get_player_rating(winner_id, surface)
        loser_rating = self.get_player_rating(loser_id, surface)

        prob_winner_wins = 1 / (
            1 + 10 ** ((loser_rating - winner_rating) / self.rating_diff_factor)
        )

        rating_change = self.k_factor * (1 - prob_winner_wins)

        self.elo_ratings[surface][winner_id] = winner_rating + rating_change
        self.elo_ratings[surface][loser_id] = loser_rating - rating_change


def _calculate_elo_ratings(match_data: pd.DataFrame, elo_config: dict) -> pd.DataFrame:
    if not elo_config:
        raise ValueError("Elo configuration ('elo_config') not found in config.yaml")

    calculator = EloCalculator(
        k_factor=elo_config["k_factor"],
        rating_diff_factor=elo_config["rating_diff_factor"],
    )

    elo_results: list[dict[str, Any]] = []

    match_data["tourney_date"] = pd.to_datetime(match_data["tourney_date"])
    match_data = match_data.sort_values(by="tourney_date").reset_index(drop=True)

    match_data["winner_historical_id"] = pd.to_numeric(
        match_data["winner_historical_id"], errors="coerce"
    )
    match_data["loser_historical_id"] = pd.to_numeric(
        match_data["loser_historical_id"], errors="coerce"
    )

    match_data.dropna(
        subset=["winner_historical_id", "loser_historical_id"], inplace=True
    )

    if match_data.empty:
        return pd.DataFrame(elo_results)

    match_data = match_data.astype(
        {"winner_historical_id": "int64", "loser_historical_id": "int64"}
    )

    match_data.drop_duplicates(subset=["match_id"], keep="first", inplace=True)

    # The 'surface' column may now contain 'Unknown', which is handled by the defaultdict
    match_data["surface"] = match_data["tourney_name"].apply(get_surface)

    for row in tqdm(
        match_data.itertuples(index=False),
        total=len(match_data),
        desc="Calculating Surface-Specific Elo",
    ):
        winner_id, loser_id, surface = (
            row.winner_historical_id,
            row.loser_historical_id,
            row.surface,
        )

        winner_pre_match_elo = calculator.get_player_rating(winner_id, surface)
        loser_pre_match_elo = calculator.get_player_rating(loser_id, surface)

        p1_id = min(winner_id, loser_id)
        p2_id = max(winner_id, loser_id)

        elo_results.append(
            {
                "match_id": row.match_id,
                "p1_id": p1_id,
                "p2_id": p2_id,
                "p1_elo": winner_pre_match_elo
                if p1_id == winner_id
                else loser_pre_match_elo,
                "p2_elo": loser_pre_match_elo
                if p1_id == winner_id
                else winner_pre_match_elo,
            }
        )

        calculator.update_ratings(
            winner_id=winner_id, loser_id=loser_id, surface=surface
        )

    return pd.DataFrame(elo_results)


def main():
    config = load_config("config.yaml")
    paths = config["data_paths"]
    elo_config = config.get("elo_config", {})

    log_info("Loading match log data for Elo calculation...")
    match_log_path = Path(paths["betfair_match_log"])
    if not match_log_path.exists():
        raise FileNotFoundError(
            f"Match log not found at {match_log_path}. "
            "Please run the 'prepare-data' command first."
        )

    df_matches = pd.read_csv(match_log_path, low_memory=False)

    df_matches.dropna(subset=["match_id", "tourney_date"], inplace=True)

    elo_df = _calculate_elo_ratings(df_matches, elo_config)

    output_path = Path(paths["elo_ratings"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if elo_df.empty:
        log_warning(
            "⚠️ No matches found to calculate Elo ratings. Saving an empty Elo file with headers."
        )
        headers = ["match_id", "p1_id", "p2_id", "p1_elo", "p2_elo"]
        pd.DataFrame(columns=headers).to_csv(output_path, index=False)
    else:
        elo_df.to_csv(output_path, index=False)

    log_info(
        f"✅ Successfully calculated and saved Surface-Specific Elo ratings to {output_path}"
    )
