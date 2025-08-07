# FILE: src/tennis_betting_model/builders/feature_builder.py
import pandas as pd
from tennis_betting_model.utils.common import get_most_recent_ranking
from tennis_betting_model.utils.config_schema import EloConfig

from tennis_betting_model.builders.feature_logic import (
    get_h2h_stats_optimized,
    get_player_stats_optimized,
)


class FeatureBuilder:
    """
    A class to build all necessary features for a given match.
    This ensures that feature generation is consistent between historical
    backtesting and live pipeline runs. It pre-processes historical data
    for fast lookups during live feature generation.
    """

    def __init__(
        self,
        player_info_lookup: dict,
        df_rankings: pd.DataFrame,
        df_matches: pd.DataFrame,
        df_elo: pd.DataFrame,
        elo_config: EloConfig,
    ):
        self.player_info_lookup = player_info_lookup
        self.df_rankings = df_rankings
        self.df_matches = df_matches.copy()
        self.df_matches["tourney_date"] = pd.to_datetime(
            self.df_matches["tourney_date"], utc=True
        )
        self.df_elo = df_elo.set_index("match_id")
        self.elo_config = elo_config

        self._prepare_data_for_lookup()

    def _prepare_data_for_lookup(self):
        """
        Pre-processes the historical match DataFrame to create indexed lookup tables,
        dramatically speeding up live feature generation.
        """
        winners = self.df_matches.copy()
        winners = winners.rename(
            columns={
                "winner_historical_id": "player_id",
                "loser_historical_id": "opponent_id",
            }
        )
        winners["won"] = 1

        losers = self.df_matches.copy()
        losers = losers.rename(
            columns={
                "loser_historical_id": "player_id",
                "winner_historical_id": "opponent_id",
            }
        )
        losers["won"] = 0

        player_match_df = pd.concat([winners, losers], ignore_index=True)
        player_match_df = player_match_df.sort_values("tourney_date")
        self.player_match_df = player_match_df.set_index("player_id")

        h2h_df = self.df_matches.copy()
        h2h_df["p1_id"] = h2h_df[["winner_historical_id", "loser_historical_id"]].min(
            axis=1
        )
        h2h_df["p2_id"] = h2h_df[["winner_historical_id", "loser_historical_id"]].max(
            axis=1
        )

        self.h2h_df = h2h_df.set_index(["p1_id", "p2_id"]).sort_index()

    def build_features(
        self,
        p1_id: int,
        p2_id: int,
        surface: str,
        match_date: pd.Timestamp,
        match_id: str,
    ) -> dict:
        """
        Constructs a feature dictionary for a single match.
        """
        p1_info = self.player_info_lookup.get(p1_id, {})
        p2_info = self.player_info_lookup.get(p2_id, {})

        p1_rank = get_most_recent_ranking(
            self.df_rankings, p1_id, match_date, self.elo_config.default_player_rank
        )
        p2_rank = get_most_recent_ranking(
            self.df_rankings, p2_id, match_date, self.elo_config.default_player_rank
        )

        try:
            match_elo = self.df_elo.loc[match_id]
            p1_elo = match_elo["p1_elo"]
            p2_elo = match_elo["p2_elo"]
        except KeyError:
            p1_elo = self.elo_config.initial_rating
            p2_elo = self.elo_config.initial_rating

        (
            p1_win_perc,
            p1_surface_win_perc,
            p1_form,
            p1_matches_last_7,
            p1_matches_last_14,
        ) = get_player_stats_optimized(self.player_match_df, p1_id, surface, match_date)
        (
            p2_win_perc,
            p2_surface_win_perc,
            p2_form,
            p2_matches_last_7,
            p2_matches_last_14,
        ) = get_player_stats_optimized(self.player_match_df, p2_id, surface, match_date)

        h2h_p1_wins, h2h_p2_wins = get_h2h_stats_optimized(
            self.h2h_df, p1_id, p2_id, match_date
        )

        feature_dict = {
            "market_id": match_id,
            "p1_id": p1_id,
            "p2_id": p2_id,
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "rank_diff": p1_rank - p2_rank,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,
            "elo_diff": p1_elo - p2_elo,
            "p1_win_perc": p1_win_perc,
            "p2_win_perc": p2_win_perc,
            "p1_surface_win_perc": p1_surface_win_perc,
            "p2_surface_win_perc": p2_surface_win_perc,
            "p1_matches_last_7_days": p1_matches_last_7,
            "p2_matches_last_7_days": p2_matches_last_7,
            "p1_matches_last_14_days": p1_matches_last_14,
            "p2_matches_last_14_days": p2_matches_last_14,
            "fatigue_diff_7_days": p1_matches_last_7 - p2_matches_last_7,
            "fatigue_diff_14_days": p1_matches_last_14 - p2_matches_last_14,
            "h2h_p1_wins": h2h_p1_wins,
            "h2h_p2_wins": h2h_p2_wins,
            "p1_hand": p1_info.get("hand", "U"),
            "p2_hand": p2_info.get("hand", "U"),
        }
        return feature_dict
