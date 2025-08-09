# FILE: src/tennis_betting_model/utils/data_loader.py
import pandas as pd
from .logger import log_info, log_error, log_success
from .schema import validate_data
from typing import Tuple, Dict, Any, cast
import glob
import os
from pathlib import Path
from functools import lru_cache
from .config_schema import DataPaths


class DataLoader:
    def __init__(self, paths: DataPaths):
        self.paths = paths

    @lru_cache(maxsize=None)
    def load_historical_player_data(self) -> pd.DataFrame:
        """Loads and consolidates all unique historical player names and IDs from raw data files."""
        raw_data_dir = Path(self.paths.raw_data_dir)
        all_players = []
        for tour in ["atp", "wta"]:
            tour_files = glob.glob(
                os.path.join(raw_data_dir, f"tennis_{tour}", f"{tour}_matches_*.csv")
            )
            if not tour_files:
                continue

            df_tour = pd.concat(
                [pd.read_csv(f, low_memory=False) for f in tour_files],
                ignore_index=True,
            )

            winners = df_tour[["winner_id", "winner_name"]].rename(
                columns={"winner_id": "historical_id", "winner_name": "historical_name"}
            )
            losers = df_tour[["loser_id", "loser_name"]].rename(
                columns={"loser_id": "historical_id", "loser_name": "historical_name"}
            )
            all_players.append(pd.concat([winners, losers]))

        if not all_players:
            return pd.DataFrame(columns=["historical_id", "historical_name"])

        df_historical = pd.concat(all_players).drop_duplicates().dropna()
        return df_historical

    @lru_cache(maxsize=None)
    def load_all_pipeline_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Any]]:
        """
        Loads, prepares, and validates all necessary data sources for the pipeline.
        This is the single source of truth for data loading.
        """
        log_info("--- Loading All Pipeline Data Sources ---")
        try:
            # Load match data
            df_matches = pd.read_csv(
                self.paths.betfair_match_log, low_memory=False, dtype={"match_id": str}
            )
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
            df_matches["winner_historical_id"] = df_matches[
                "winner_historical_id"
            ].astype(int)
            df_matches["loser_historical_id"] = df_matches[
                "loser_historical_id"
            ].astype(int)
            df_matches = validate_data(
                df_matches, "betfair_match_log", "Betfair Match Log"
            )

            # Load player data
            df_players = pd.read_csv(self.paths.raw_players, encoding="latin-1")
            df_players["player_id"] = pd.to_numeric(
                df_players["player_id"], errors="coerce"
            )
            df_players.dropna(subset=["player_id"], inplace=True)
            df_players["player_id"] = df_players["player_id"].astype(int)
            df_players = df_players.drop_duplicates(subset=["player_id"], keep="first")
            player_info_lookup = df_players.set_index("player_id").to_dict("index")
            validate_data(df_players, "raw_players", "Raw Player Attributes")

            # Load rankings data
            df_rankings = pd.read_csv(self.paths.consolidated_rankings)
            df_rankings["ranking_date"] = pd.to_datetime(
                df_rankings["ranking_date"], utc=True
            )
            df_rankings = df_rankings.sort_values(by="ranking_date")
            validate_data(df_rankings, "consolidated_rankings", "Consolidated Rankings")

            # Load Elo ratings data
            df_elo = pd.read_csv(self.paths.elo_ratings, dtype={"match_id": str})

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

    @lru_cache(maxsize=None)
    def load_backtest_data_for_dashboard(self) -> pd.DataFrame:
        """Loads and prepares the backtest results data specifically for the dashboard."""
        try:
            results_path = Path(self.paths.backtest_results)
            df = pd.read_csv(results_path, dtype={"market_id": str})
            df["tourney_date"] = pd.to_datetime(df["tourney_date"])

            if "pnl" not in df.columns:
                df["pnl"] = df.apply(
                    lambda row: (row["odds"] - 1) * 0.95 if row["winner"] == 1 else -1,
                    axis=1,
                )

            features_path = Path(self.paths.consolidated_features)
            if features_path.exists():
                df_features = pd.read_csv(
                    features_path,
                    usecols=["market_id", "rank_diff"],
                    dtype={"market_id": str},
                )
                df = pd.merge(df, df_features, on="market_id", how="left")
            else:
                df["rank_diff"] = 0

            return df.sort_values("tourney_date")
        except FileNotFoundError:
            log_error(
                f"Backtest results not found at {self.paths.backtest_results}. Please run a backtest first."
            )
            return pd.DataFrame()
        except Exception as e:
            log_error(f"An error occurred loading backtest data: {e}")
            return pd.DataFrame()
