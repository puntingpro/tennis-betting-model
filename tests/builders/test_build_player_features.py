# tests/builders/test_build_player_features.py
import pandas as pd
import pytest
from src.scripts.builders.build_player_features import (
    calculate_player_stats,
    add_h2h_stats,
)


# --- Unchanged Fixture ---
@pytest.fixture
def sample_match_data() -> pd.DataFrame:
    # ... (content remains the same)
    data = {
        "match_id": [1, 2, 3, 4],
        "tourney_date": pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        ),
        "surface": ["Hard", "Hard", "Clay", "Hard"],
        "winner_id": [101, 102, 101, 103],
        "loser_id": [102, 103, 103, 101],
    }
    return pd.DataFrame(data)


# --- Unchanged Fixture ---
@pytest.fixture
def detailed_match_data() -> pd.DataFrame:
    # ... (content remains the same)
    data = {
        "match_id": range(1, 13),
        "tourney_date": pd.to_datetime([f"2023-01-{i:02d}" for i in range(1, 13)]),
        "surface": [
            "Hard",
            "Hard",
            "Clay",
            "Clay",
            "Grass",
            "Grass",
            "Hard",
            "Hard",
            "Clay",
            "Hard",
            "Hard",
            "Hard",
        ],
        "winner_id": [101, 101, 102, 101, 102, 101, 101, 102, 101, 101, 101, 101],
        "loser_id": [102, 103, 101, 103, 101, 103, 102, 101, 102, 103, 102, 103],
    }
    return pd.DataFrame(data)


# --- Unchanged Test ---
def test_calculate_player_stats_win_percentages(sample_match_data):
    # ... (content remains the same)
    stats_df = calculate_player_stats(sample_match_data)
    p101_stats = stats_df[stats_df["player_id"] == 101].sort_values("match_id")
    assert p101_stats[p101_stats["match_id"] == 4]["win_perc"].iloc[0] == 1.0
    p103_stats = stats_df[stats_df["player_id"] == 103].sort_values("match_id")
    assert p103_stats[p103_stats["match_id"] == 4]["surface_win_perc"].iloc[0] == 0.0
    assert p103_stats[p103_stats["match_id"] == 4]["win_perc"].iloc[0] == 0.0


# --- MODIFIED: Corrected the test logic and assertion ---
def test_calculate_player_stats_form_and_first_match(detailed_match_data):
    """
    Tests rolling form and ensures stats are zero for a player's first match.
    """
    stats_df = calculate_player_stats(detailed_match_data)

    # Player 102's stats
    p102_stats = stats_df[stats_df["player_id"] == 102].sort_values("match_id")

    # Check stats before their first match (match_id 1)
    first_match = p102_stats[p102_stats["match_id"] == 1].iloc[0]
    assert first_match["win_perc"] == 0.0
    assert first_match["surface_win_perc"] == 0.0
    assert first_match["form_last_10"] == 0.0

    # Check form before their match on Jan 11 (match_id 11)
    # Player 102's record before this match is 3 wins and 3 losses. (3/6 = 0.5)
    form_check = p102_stats[p102_stats["match_id"] == 11]["form_last_10"].iloc[0]
    assert pytest.approx(form_check) == 0.5


# --- Unchanged Fixtures and Tests ---
@pytest.fixture
def sample_h2h_data() -> pd.DataFrame:
    # ... (content remains the same)
    data = {
        "match_id": ["m1", "m2", "m3", "m4"],
        "tourney_date": pd.to_datetime(
            ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]
        ),
        "p1_id": [101, 101, 102, 103],
        "p2_id": [102, 102, 103, 101],
        "winner_id": [101, 102, 102, 101],
    }
    return pd.DataFrame(data)


def test_add_h2h_stats(sample_h2h_data):
    # ... (content remains the same)
    h2h_df = add_h2h_stats(sample_h2h_data)
    match1 = h2h_df[h2h_df["match_id"] == "m1"].iloc[0]
    assert match1["h2h_p1_wins"] == 0
    assert match1["h2h_p2_wins"] == 0
    match2 = h2h_df[h2h_df["match_id"] == "m2"].iloc[0]
    assert match2["h2h_p1_wins"] == 1
    assert match2["h2h_p2_wins"] == 0
    match4 = h2h_df[h2h_df["match_id"] == "m4"].iloc[0]
    assert match4["h2h_p1_wins"] == 0
    assert match4["h2h_p2_wins"] == 0


def test_add_h2h_stats_multiple_meetings(sample_h2h_data):
    # ... (content remains the same)
    extra_match = pd.DataFrame(
        {
            "match_id": ["m5"],
            "tourney_date": pd.to_datetime(["2023-05-01"]),
            "p1_id": [101],
            "p2_id": [102],
            "winner_id": [101],
        }
    )
    extended_h2h_data = pd.concat([sample_h2h_data, extra_match], ignore_index=True)
    h2h_df = add_h2h_stats(extended_h2h_data)
    match5 = h2h_df[h2h_df["match_id"] == "m5"].iloc[0]
    assert match5["h2h_p1_wins"] == 1
    assert match5["h2h_p2_wins"] == 1
