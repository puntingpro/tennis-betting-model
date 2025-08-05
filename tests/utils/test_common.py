# tests/utils/test_common.py

import pytest
import pandas as pd
import numpy as np
from tennis_betting_model.utils.common import (
    get_surface,
    get_tournament_category,
    normalize_df_column_names,
    patch_winner_column,
)


@pytest.mark.parametrize(
    "tourney_name, expected_surface",
    [
        ("Australian Open", "Hard"),
        ("Roland Garros", "Clay"),
        ("Wimbledon", "Grass"),
        ("ATP Masters 1000 Miami", "Hard"),
        ("ATP Challenger Oeiras 2, Portugal (Clay)", "Clay"),
        ("ITF M15 Monastir (Grass)", "Grass"),
        ("Some Random Event", "Hard"),
    ],
)
def test_get_surface(tourney_name, expected_surface):
    """Tests that surfaces are correctly identified from tournament names."""
    assert get_surface(tourney_name) == expected_surface


@pytest.mark.parametrize(
    "tourney_name, expected_category",
    [
        ("Australian Open", "Grand Slam"),
        ("Wimbledon", "Grand Slam"),
        ("ATP Masters 1000 Rome", "Masters 1000"),
        ("WTA Tour Finals", "Tour Finals"),
        ("Davis Cup", "Team Event"),
        ("ATP Challenger Seville", "Challenger"),
        ("ITF M15 Sharm El Sheikh", "ITF / Futures"),
        ("ATP 250 Doha", "ATP / WTA Tour"),
    ],
)
def test_get_tournament_category(tourney_name, expected_category):
    """Tests that tournament categories are correctly identified."""
    assert get_tournament_category(tourney_name) == expected_category


def test_normalize_df_column_names():
    """Tests that DataFrame column names are correctly normalized."""
    df = pd.DataFrame(
        columns=[
            "Player Name",
            "Expected Value (EV)",
            "rank",
            "market_id",
        ]
    )
    normalized_df = normalize_df_column_names(df)
    expected_columns = [
        "player_name",
        "expected_value_ev",
        "rank",
        "market_id",
    ]
    assert list(normalized_df.columns) == expected_columns


def test_patch_winner_column_adds_column_if_missing():
    """Tests that the 'winner' column is added and filled with 0 if it does not exist."""
    df = pd.DataFrame({"odds": [1.5, 2.5]})
    patched_df = patch_winner_column(df)
    assert "winner" in patched_df.columns
    assert patched_df["winner"].equals(pd.Series([0, 0], dtype=int))


def test_patch_winner_column_handles_mixed_types_and_nans():
    """Tests that an existing 'winner' column with mixed types or NaNs is correctly patched."""
    df = pd.DataFrame({"winner": [1.0, np.nan, 0, 1]})
    patched_df = patch_winner_column(df)
    expected = pd.Series([1, 0, 0, 1], dtype=int)
    assert patched_df["winner"].equals(expected)
