# tests/builders/test_feature_builder.py

import pandas as pd
import pytest
from tennis_betting_model.builders.feature_builder import FeatureBuilder
from tennis_betting_model.utils.config_schema import EloConfig


@pytest.fixture
def mock_builder_data():
    """Provides mock dataframes needed to instantiate the FeatureBuilder."""
    player_info = {101: {"hand": "R"}, 102: {"hand": "L"}}

    df_rankings = pd.DataFrame(
        {
            "ranking_date": pd.to_datetime(["2023-01-01"], utc=True),
            "player": [101],
            "rank": [10],
        }
    ).sort_values(by="ranking_date")

    df_matches = pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(["2023-01-05"], utc=True),
            "surface": ["Hard"],
            "winner_historical_id": [101],
            "loser_historical_id": [102],
            # --- FIX: Add the missing 'sets_played' column to the mock data ---
            "sets_played": [3],
        }
    )

    df_elo = pd.DataFrame(
        {
            "match_id": ["m1"],
            "p1_id": [101],
            "p2_id": [102],
            "p1_elo": [1600],
            "p2_elo": [1550],
        }
    )

    elo_config = EloConfig(
        k_factor=32,
        rating_diff_factor=400,
        initial_rating=1500,
        default_player_rank=500,
    )

    return player_info, df_rankings, df_matches, df_elo, elo_config


def test_feature_builder_builds_correct_features(mock_builder_data):
    """
    Tests that the unified FeatureBuilder correctly calculates and assembles a
    complete feature dictionary for a given match.
    """
    player_info, df_rankings, df_matches, df_elo, elo_config = mock_builder_data

    builder = FeatureBuilder(
        player_info_lookup=player_info,
        df_rankings=df_rankings,
        df_matches=df_matches,
        df_elo=df_elo,
        elo_config=elo_config,
    )

    features = builder.build_features(
        p1_id=101,
        p2_id=102,
        surface="Hard",
        match_date=pd.to_datetime("2023-01-10", utc=True),
        match_id="m1",
    )

    # Check some key features
    assert features["p1_id"] == 101
    assert features["p1_rank"] == 10
    assert features["p2_rank"] == 500  # Default rank for player with no history
    assert features["rank_diff"] == -490

    assert features["p1_elo"] == 1600
    assert features["p2_elo"] == 1550
    assert features["elo_diff"] == 50

    # From the single match history, p1 won
    assert features["p1_win_perc"] == 1.0
    assert features["p1_surface_win_perc"] == 1.0
    assert features["p2_win_perc"] == 0.0

    # The match was on 2023-01-05, current date is 2023-01-10
    assert features["p1_matches_last_7_days"] == 1
    assert features["p1_matches_last_14_days"] == 1
    assert features["p2_matches_last_7_days"] == 1

    assert features["p1_hand"] == "R"
