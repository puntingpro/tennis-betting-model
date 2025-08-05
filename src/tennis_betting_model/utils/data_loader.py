# FILE: src/tennis_betting_model/utils/data_loader.py
import pandas as pd
from .logger import log_info, log_error, log_success
from .schema import validate_data
from typing import Tuple, Dict, Any, cast


def load_all_pipeline_data(
    paths: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Any]]:
    """
    Loads, prepares, and validates all necessary data sources for the pipeline.
    This is the single source of truth for data loading.
    """
    log_info("--- Loading All Pipeline Data Sources ---")
    try:
        # Load match data
        df_matches = pd.read_csv(paths["betfair_match_log"], low_memory=False)
        df_matches["tourney_date"] = pd.to_datetime(
            df_matches["tourney_date"], errors="coerce", utc=True
        )
        df_matches["winner_historical_id"] = pd.to_numeric(
            df_matches["winner_historical_id"], errors="coerce"
        )
        df_matches["loser_historical_id"] = pd.to_numeric(
            df_matches["loser_historical_id"], errors="coerce"
        )
        df_matches.dropna(
            subset=["tourney_date", "winner_historical_id", "loser_historical_id"],
            inplace=True,
        )
        df_matches["winner_historical_id"] = df_matches["winner_historical_id"].astype(
            int
        )
        df_matches["loser_historical_id"] = df_matches["loser_historical_id"].astype(
            int
        )
        df_matches["match_id"] = df_matches["match_id"].astype(str)
        df_matches = validate_data(df_matches, "betfair_match_log", "Betfair Match Log")

        # Load player data
        df_players = pd.read_csv(paths["raw_players"], encoding="latin-1")
        df_players["player_id"] = pd.to_numeric(
            df_players["player_id"], errors="coerce"
        )
        df_players.dropna(subset=["player_id"], inplace=True)
        df_players["player_id"] = df_players["player_id"].astype(int)
        df_players = df_players.drop_duplicates(subset=["player_id"], keep="first")
        player_info_lookup = df_players.set_index("player_id").to_dict("index")
        validate_data(df_players, "raw_players", "Raw Player Attributes")

        # Load rankings data
        df_rankings = pd.read_csv(paths["consolidated_rankings"])
        df_rankings["ranking_date"] = pd.to_datetime(
            df_rankings["ranking_date"], utc=True
        )
        df_rankings = df_rankings.sort_values(by="ranking_date")
        validate_data(df_rankings, "consolidated_rankings", "Consolidated Rankings")

        # Load Elo ratings data
        df_elo = pd.read_csv(paths["elo_ratings"])
        df_elo["match_id"] = df_elo["match_id"].astype(str)

        log_success("✅ All data loaded and validated successfully.")
        return (
            df_matches,
            df_rankings,
            df_players,
            df_elo,
            cast(Dict[int, Any], player_info_lookup),
        )

    except FileNotFoundError as e:
        log_error(f"A required data file was not found. Error: {e}")
        raise
    except Exception as e:
        log_error(f"An unexpected error occurred during data loading: {e}")
        raise
