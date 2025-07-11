# src/scripts/builders/build_elo_ratings.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.scripts.utils.config import load_config
from src.scripts.utils.logger import setup_logging, log_info, log_success

# Constants for the Elo calculation
K_FACTOR = 32
INITIAL_RATING = 1500

def calculate_expected_score(rating1, rating2):
    """Calculates the expected score of player 1 against player 2."""
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_elo(winner_rating, loser_rating):
    """Updates the Elo ratings for a winner and a loser."""
    expected_win = calculate_expected_score(winner_rating, loser_rating)
    
    new_winner_rating = winner_rating + K_FACTOR * (1 - expected_win)
    new_loser_rating = loser_rating + K_FACTOR * (0 - (1 - expected_win))
    
    return new_winner_rating, new_loser_rating

def generate_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates point-in-time Elo ratings for all matches.
    """
    log_info("Generating historical Elo ratings for all players...")
    
    elo_ratings = {}
    elo_history = []

    # Sort matches chronologically to process them in order
    df = df.sort_values(by='tourney_date').reset_index(drop=True)

    for row in tqdm(df.itertuples(), total=len(df), desc="Calculating Elo"):
        winner_id = row.winner_id
        loser_id = row.loser_id

        # Get the current ratings, defaulting to the initial rating if a player is new
        winner_elo = elo_ratings.get(winner_id, INITIAL_RATING)
        loser_elo = elo_ratings.get(loser_id, INITIAL_RATING)

        # Store the ratings *before* the match
        elo_history.append({
            'match_id': row.match_id,
            'p1_id_elo': winner_id,
            'p2_id_elo': loser_id,
            'p1_elo': winner_elo,
            'p2_elo': loser_elo
        })

        # Calculate the new ratings after the match
        new_winner_elo, new_loser_elo = update_elo(winner_elo, loser_elo)

        # Update the ratings in our lookup
        elo_ratings[winner_id] = new_winner_elo
        elo_ratings[loser_id] = new_loser_elo

    return pd.DataFrame(elo_history)

def main():
    """Main CLI entrypoint for building Elo ratings."""
    setup_logging()
    config = load_config("config.yaml")
    paths = config['data_paths']

    log_info(f"Loading consolidated match data from {paths['consolidated_matches']}...")
    df_matches = pd.read_csv(paths['consolidated_matches'])
    
    df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'], errors='coerce')
    df_matches.dropna(subset=['tourney_date', 'winner_id', 'loser_id'], inplace=True)
    
    # Ensure match_id is present, creating it if necessary
    if 'match_id' not in df_matches.columns:
        df_matches['match_id'] = df_matches['tourney_id'].astype(str) + '-' + df_matches['match_num'].astype(str)

    elo_df = generate_elo_ratings(df_matches)

    output_path = Path(paths['elo_ratings'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    elo_df.to_csv(output_path, index=False)
    log_success(f"✅ Successfully saved Elo ratings to {output_path}")

if __name__ == "__main__":
    main()