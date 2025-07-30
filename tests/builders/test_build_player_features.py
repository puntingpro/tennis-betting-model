# tests/builders/test_build_player_features.py

import pandas as pd
import pytest
from tennis_betting_model.builders.build_player_features import calculate_player_stats


@pytest.fixture
def mock_match_history():
    """Creates a small, controlled match history DataFrame with dates."""
    data = {
        "match_id": ["m1", "m2", "m3", "m4"],
        "tourney_name": ["A", "B", "A", "B"],
        "surface": ["Hard", "Clay", "Hard", "Clay"],
        "tourney_date": pd.to_datetime(
            ["2023-01-01", "2023-01-08", "2023-01-10", "2023-01-20"]
        ),
        "winner_historical_id": [101, 102, 101, 103],
        "loser_historical_id": [102, 103, 103, 101],
    }
    df = pd.DataFrame(data)
    # Ensure correct types for processing
    df["winner_historical_id"] = df["winner_historical_id"].astype("Int64")
    df["loser_historical_id"] = df["loser_historical_id"].astype("Int64")
    return df


def test_calculate_player_stats_win_perc(mock_match_history):
    """
    Tests that win percentages are calculated point-in-time (anti-leakage).
    """
    player_stats_df = calculate_player_stats(mock_match_history)

    # For match 'm3' (played 2023-01-10), player 101 has played one match (m1) and won it.
    stats_m3_p101 = player_stats_df[
        (player_stats_df["match_id"] == "m3") & (player_stats_df["player_id"] == 101)
    ]
    assert stats_m3_p101.iloc[0]["win_perc"] == 1.0  # 1 win / 1 match


def test_calculate_player_fatigue_features(mock_match_history):
    """
    Tests that fatigue features (rolling match counts) are calculated correctly.
    """
    player_stats_df = calculate_player_stats(mock_match_history)

    # For match 'm4' (played 2023-01-20), check player 101's fatigue
    stats_m4_p101 = player_stats_df[
        (player_stats_df["match_id"] == "m4") & (player_stats_df["player_id"] == 101)
    ]
    # Prior to m4, player 101 played on Jan 1 and Jan 10.
    # 7-day window (Jan 14-20): No matches.
    # 14-day window (Jan 7-20): One match on Jan 10.
    assert stats_m4_p101.iloc[0]["matches_last_7_days"] == 0
    assert stats_m4_p101.iloc[0]["matches_last_14_days"] == 1


def test_calculate_stats_for_new_player(mock_match_history):
    """
    Tests that a player appearing for the first time has default stats.
    """
    player_stats_df = calculate_player_stats(mock_match_history)

    # In match 'm1' (played first), it's the first match for both 101 and 102.
    # Their stats going into this match should all be 0.
    stats_m1_p101 = player_stats_df[
        (player_stats_df["match_id"] == "m1") & (player_stats_df["player_id"] == 101)
    ]

    assert stats_m1_p101.iloc[0]["win_perc"] == 0.0
    assert stats_m1_p101.iloc[0]["surface_win_perc"] == 0.0
    assert stats_m1_p101.iloc[0]["matches_last_7_days"] == 0
    assert stats_m1_p101.iloc[0]["matches_last_14_days"] == 0
