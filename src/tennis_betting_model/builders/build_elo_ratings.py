# src/tennis_betting_model/builders/build_elo_ratings.py

from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tennis_betting_model.utils.config import load_config


@dataclass
class EloCalculator:
    k_factor: int
    rating_diff_factor: int
    initial_rating: int = 1500
    elo_ratings: dict[int, float] = field(default_factory=dict)

    def get_player_rating(self, player_id: int) -> float:
        return self.elo_ratings.get(player_id, self.initial_rating)

    def update_ratings(self, winner_id: int, loser_id: int) -> None:
        winner_rating = self.get_player_rating(winner_id)

        loser_rating = self.get_player_rating(loser_id)

        prob_winner_wins = 1 / (
            1 + 10 ** ((loser_rating - winner_rating) / self.rating_diff_factor)
        )
        self.elo_ratings[winner_id] = winner_rating + self.k_factor * (
            1 - prob_winner_wins
        )
        self.elo_ratings[loser_id] = loser_rating - self.k_factor * (prob_winner_wins)


def _calculate_elo_ratings(match_data: pd.DataFrame, elo_config: dict) -> pd.DataFrame:
    if not elo_config:
        raise ValueError("Elo configuration ('elo_config') not found in config.yaml")

    calculator = EloCalculator(
        k_factor=elo_config["k_factor"],
        rating_diff_factor=elo_config["rating_diff_factor"],
    )

    elo_results = []

    match_data["tourney_date"] = pd.to_datetime(match_data["tourney_date"])
    match_data = match_data.sort_values(by="tourney_date").reset_index(drop=True)

    # --- FIX: Add robust data cleaning to handle corrupted input ---
    # Coerce non-numeric values in ID columns to NaN (Not a Number)

    match_data["winner_historical_id"] = pd.to_numeric(
        match_data["winner_historical_id"], errors="coerce"
    )
    match_data["loser_historical_id"] = pd.to_numeric(
        match_data["loser_historical_id"], errors="coerce"
    )

    # Drop rows where IDs are missing, as they are unusable for Elo
    match_data.dropna(
        subset=["winner_historical_id", "loser_historical_id"], inplace=True
    )

    # Now that the data is clean, safely convert to integer
    match_data = match_data.astype(
        {"winner_historical_id": "int64", "loser_historical_id": "int64"}
    )
    # --- END FIX ---

    for row in tqdm(
        match_data.itertuples(index=False),
        total=len(match_data),
        desc="Calculating Elo Ratings",
    ):
        winner_id, loser_id = row.winner_historical_id, row.loser_historical_id

        winner_pre_match_elo = calculator.get_player_rating(winner_id)
        loser_pre_match_elo = calculator.get_player_rating(loser_id)

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

        calculator.update_ratings(winner_id=winner_id, loser_id=loser_id)

    return pd.DataFrame(elo_results)


def main():
    config = load_config("config.yaml")
    paths = config["data_paths"]
    elo_config = config.get("elo_config", {})

    print("Loading match log data for Elo calculation...")
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
    elo_df.to_csv(output_path, index=False)
    print(f"✅ Successfully calculated and saved Elo ratings to {output_path}")
