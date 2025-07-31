import pandas as pd
from tennis_betting_model.utils.common import get_most_recent_ranking
from tennis_betting_model.utils.constants import ELO_INITIAL_RATING

from tennis_betting_model.builders.feature_logic import (
    get_h2h_stats,
    get_player_stats,
)


class FeatureBuilder:
    """
    A class to build all necessary features for a given match.
    This ensures that feature generation is consistent between historical
    backtesting and live pipeline runs.
    """

    def __init__(
        self,
        player_info_lookup: dict,
        df_rankings: pd.DataFrame,
        df_matches: pd.DataFrame,
        df_elo: pd.DataFrame,
    ):
        self.player_info_lookup = player_info_lookup
        self.df_rankings = df_rankings

        # --- FINAL FIX: Force df_matches tourney_date to be UTC-aware on initialization ---
        self.df_matches = df_matches.copy()
        self.df_matches["tourney_date"] = pd.to_datetime(
            self.df_matches["tourney_date"], utc=True
        )

        self.df_elo = df_elo.set_index("match_id")

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

        # Get point-in-time rankings
        p1_rank = get_most_recent_ranking(self.df_rankings, p1_id, match_date)
        p2_rank = get_most_recent_ranking(self.df_rankings, p2_id, match_date)

        # Get point-in-time Elo
        try:
            match_elo = self.df_elo.loc[match_id]
            p1_elo = match_elo["p1_elo"]
            p2_elo = match_elo["p2_elo"]
        except KeyError:
            p1_elo = ELO_INITIAL_RATING
            p2_elo = ELO_INITIAL_RATING

        # Get form, win percentages, and fatigue
        (
            p1_win_perc,
            p1_surface_win_perc,
            p1_form,
            p1_matches_last_7,
            p1_matches_last_14,
        ) = get_player_stats(self.df_matches, p1_id, surface, match_date)
        (
            p2_win_perc,
            p2_surface_win_perc,
            p2_form,
            p2_matches_last_7,
            p2_matches_last_14,
        ) = get_player_stats(self.df_matches, p2_id, surface, match_date)

        # Get H2H stats
        h2h_p1_wins, h2h_p2_wins = get_h2h_stats(
            self.df_matches, p1_id, p2_id, match_date
        )

        # Assemble feature dictionary
        feature_dict = {
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
