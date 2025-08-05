# FILE: src/tennis_betting_model/utils/data_loader.py
import pandas as pd
from .logger import log_info


def load_pipeline_data(paths: dict) -> tuple:
    """Loads all necessary data sources for the pipeline."""
    log_info("Loading model and all required data sources...")

    # FIX: Add low_memory=False to suppress DtypeWarning and ensure UTC timezone awareness
    df_matches = pd.read_csv(paths["betfair_match_log"], low_memory=False)
    df_matches["tourney_date"] = pd.to_datetime(
        df_matches["tourney_date"], errors="coerce", utc=True
    )

    # Ensure IDs are numeric and handle potential errors during loading
    df_matches["winner_historical_id"] = pd.to_numeric(
        df_matches["winner_historical_id"], errors="coerce"
    )
    df_matches["loser_historical_id"] = pd.to_numeric(
        df_matches["loser_historical_id"], errors="coerce"
    )

    # Drop rows where critical data (date or IDs) is missing after coercion
    df_matches.dropna(
        subset=["tourney_date", "winner_historical_id", "loser_historical_id"],
        inplace=True,
    )

    df_matches["winner_historical_id"] = df_matches["winner_historical_id"].astype(int)
    df_matches["loser_historical_id"] = df_matches["loser_historical_id"].astype(int)

    df_players = pd.read_csv(paths["raw_players"], encoding="latin-1")

    # Clean player data IDs
    df_players["player_id"] = pd.to_numeric(df_players["player_id"], errors="coerce")
    df_players.dropna(subset=["player_id"], inplace=True)
    df_players["player_id"] = df_players["player_id"].astype(int)

    df_players = df_players.drop_duplicates(subset=["player_id"], keep="first")
    player_info_lookup = df_players.set_index("player_id").to_dict("index")

    df_rankings = pd.read_csv(paths["consolidated_rankings"])
    # Ensure rankings are also UTC
    df_rankings["ranking_date"] = pd.to_datetime(df_rankings["ranking_date"], utc=True)
    df_rankings = df_rankings.sort_values(by="ranking_date")

    # Load the Elo ratings data for the live pipeline
    df_elo = pd.read_csv(paths["elo_ratings"])

    log_info("✅ All data loaded successfully.")
    return player_info_lookup, df_rankings, df_matches, df_elo
