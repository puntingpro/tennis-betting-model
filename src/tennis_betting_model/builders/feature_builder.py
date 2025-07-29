import pandas as pd
from tennis_betting_model.utils.common import get_most_recent_ranking
from tennis_betting_model.pipeline.feature_engineering import (
    get_h2h_stats,
    get_player_form_and_win_perc,
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
    ):
        self.player_info_lookup = player_info_lookup
        self.df_rankings = df_rankings
        self.df_matches = df_matches

    def build_features(
        self, p1_id: int, p2_id: int, surface: str, match_date: pd.Timestamp
    ) -> dict:
        """
        Constructs a feature dictionary for a single match.
        """
        p1_info = self.player_info_lookup.get(p1_id, {})
        p2_info = self.player_info_lookup.get(p2_id, {})

        # Get point-in-time rankings
        p1_rank = get_most_recent_ranking(self.df_rankings, p1_id, match_date)
        p2_rank = get_most_recent_ranking(self.df_rankings, p2_id, match_date)

        # Get form and win percentages
        p1_win_perc, p1_surface_win_perc, p1_form = get_player_form_and_win_perc(
            self.df_matches, p1_id, surface, match_date
        )
        p2_win_perc, p2_surface_win_perc, p2_form = get_player_form_and_win_perc(
            self.df_matches, p2_id, surface, match_date
        )

        # Get H2H stats
        h2h_p1_wins, h2h_p2_wins = get_h2h_stats(
            self.df_matches, p1_id, p2_id, match_date
        )
        h2h_total = h2h_p1_wins + h2h_p2_wins
        h2h_win_perc_p1 = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5

        # Assemble feature dictionary
        feature_dict = {
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "rank_diff": p1_rank - p2_rank
            if pd.notna(p1_rank) and pd.notna(p2_rank)
            else 0,
            "p1_height": p1_info.get("height", 180),  # Default height
            "p2_height": p2_info.get("height", 180),
            "h2h_p1_wins": h2h_p1_wins,
            "h2h_p2_wins": h2h_p2_wins,
            "h2h_win_perc_p1": h2h_win_perc_p1,
            "p1_win_perc": p1_win_perc,
            "p2_win_perc": p2_win_perc,
            "p1_surface_win_perc": p1_surface_win_perc,
            "p2_surface_win_perc": p2_surface_win_perc,
            "p1_form_last_10": p1_form,
            "p2_form_last_10": p2_form,
            "p1_hand_L": 1 if p1_info.get("hand") == "L" else 0,
            "p1_hand_R": 1 if p1_info.get("hand") == "R" else 0,
            "p1_hand_U": 1 if p1_info.get("hand") == "U" else 0,
            "p2_hand_L": 1 if p2_info.get("hand") == "L" else 0,
            "p2_hand_R": 1 if p2_info.get("hand") == "R" else 0,
            "p2_hand_U": 1 if p2_info.get("hand") == "U" else 0,
        }
        return feature_dict
