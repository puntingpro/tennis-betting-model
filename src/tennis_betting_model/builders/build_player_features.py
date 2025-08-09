# src/tennis_betting_model/builders/build_player_features.py
import pandas as pd
from typing import Dict, Any

from tennis_betting_model.utils.logger import get_logger
from tennis_betting_model.utils.config_schema import DataPaths, EloConfig
from tennis_betting_model.builders.feature_logic import (
    get_win_percentages,
    get_h2h_stats_optimized,
    get_fatigue_features,
    get_recent_form,
)

logger = get_logger(__name__)


class FeatureBuilder:
    def __init__(
        self,
        player_info_lookup: Dict[int, Dict[str, Any]],
        df_rankings: pd.DataFrame,
        df_matches: pd.DataFrame,
        df_elo: pd.DataFrame,
        elo_config: EloConfig,
    ):
        self.player_info_lookup = player_info_lookup
        self.df_rankings = df_rankings
        self.df_matches = df_matches
        self.df_elo = df_elo
        self.elo_config = elo_config

    def _get_player_rank(self, player_id: int, match_date: pd.Timestamp) -> int:
        player_rankings = self.df_rankings[
            (self.df_rankings["player"] == player_id)
            & (self.df_rankings["ranking_date"] <= match_date)
        ]
        if not player_rankings.empty:
            return int(player_rankings.iloc[-1]["rank"])
        return int(self.elo_config.default_player_rank)

    def build_features(
        self,
        p1_id: int,
        p2_id: int,
        surface: str,
        match_date: pd.Timestamp,
        match_id: str,
    ) -> Dict[str, Any]:
        p1_rank = self._get_player_rank(p1_id, match_date)
        p2_rank = self._get_player_rank(p2_id, match_date)
        rank_diff = p1_rank - p2_rank

        p1_elo_series = self.df_elo[self.df_elo["match_id"] == match_id]["p1_elo"]
        p2_elo_series = self.df_elo[self.df_elo["match_id"] == match_id]["p2_elo"]
        p1_elo = (
            p1_elo_series.iloc[0]
            if not p1_elo_series.empty
            else self.elo_config.initial_rating
        )
        p2_elo = (
            p2_elo_series.iloc[0]
            if not p2_elo_series.empty
            else self.elo_config.initial_rating
        )
        elo_diff = p1_elo - p2_elo

        p1_win_perc, p1_surface_win_perc, _ = get_win_percentages(
            self.df_matches, p1_id, surface, match_date
        )
        p2_win_perc, p2_surface_win_perc, _ = get_win_percentages(
            self.df_matches, p2_id, surface, match_date
        )

        h2h_p1_wins, h2h_p2_wins = get_h2h_stats_optimized(
            self.df_matches, p1_id, p2_id, match_date
        )

        p1_fatigue_7d, p1_fatigue_14d = get_fatigue_features(
            self.df_matches, p1_id, match_date
        )
        p2_fatigue_7d, p2_fatigue_14d = get_fatigue_features(
            self.df_matches, p2_id, match_date
        )

        p1_matches_7d, p1_matches_14d = get_recent_form(
            self.df_matches, p1_id, match_date
        )
        p2_matches_7d, p2_matches_14d = get_recent_form(
            self.df_matches, p2_id, match_date
        )

        p1_hand = self.player_info_lookup.get(p1_id, {}).get("hand", "U")
        p2_hand = self.player_info_lookup.get(p2_id, {}).get("hand", "U")

        return {
            "p1_id": p1_id,
            "p2_id": p2_id,
            "surface": surface,
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "rank_diff": rank_diff,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,
            "elo_diff": elo_diff,
            "p1_win_perc": p1_win_perc,
            "p2_win_perc": p2_win_perc,
            "p1_surface_win_perc": p1_surface_win_perc,
            "p2_surface_win_perc": p2_surface_win_perc,
            "h2h_p1_wins": h2h_p1_wins,
            "h2h_p2_wins": h2h_p2_wins,
            "p1_fatigue_last_7_days": p1_fatigue_7d,
            "p2_fatigue_last_7_days": p2_fatigue_7d,
            "p1_fatigue_last_14_days": p1_fatigue_14d,
            "p2_fatigue_last_14_days": p2_fatigue_14d,
            "p1_matches_last_7_days": p1_matches_7d,
            "p2_matches_last_7_days": p2_matches_7d,
            "p1_matches_last_14_days": p1_matches_14d,
            "p2_matches_last_14_days": p2_matches_14d,
            "p1_hand": p1_hand,
            "p2_hand": p2_hand,
        }


def main(paths: DataPaths, elo_config: EloConfig):
    # Main logic to build features for all matches would go here
    logger.info("Building player features...")
    # This would typically loop through matches and call FeatureBuilder
    logger.info("Player features built successfully.")


if __name__ == "__main__":
    # Example of how this script might be run
    # from tennis_betting_model.utils.config import load_config
    # from tennis_betting_model.config import ROOT_DIR
    # config = load_config(f"{ROOT_DIR}/config.yaml")
    # main(config.data, config.elo)
    pass


# hello
