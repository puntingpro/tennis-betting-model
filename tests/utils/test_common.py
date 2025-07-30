# tests/utils/test_common.py

import pytest
from tennis_betting_model.utils.common import get_surface, get_tournament_category


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
