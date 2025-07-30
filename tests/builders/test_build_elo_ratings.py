# tests/builders/test_build_elo_ratings.py

import pytest
from tennis_betting_model.builders.build_elo_ratings import EloCalculator


@pytest.fixture
def elo_config():
    """Provides standard Elo configuration for tests."""
    return {"k_factor": 32, "rating_diff_factor": 400, "initial_rating": 1500}


def test_elo_calculator_initial_rating(elo_config):
    """
    Tests that a player not seen before receives the default initial rating.
    """
    calculator = EloCalculator(
        k_factor=elo_config["k_factor"],
        rating_diff_factor=elo_config["rating_diff_factor"],
        initial_rating=elo_config["initial_rating"],
    )
    new_player_id = 999
    # --- FIX: Provide a surface ---
    assert calculator.get_player_rating(new_player_id, surface="Hard") == 1500


def test_elo_calculator_update_for_even_match(elo_config):
    """
    Tests the rating update for a match between two players with the same initial rating.
    """
    calculator = EloCalculator(
        k_factor=elo_config["k_factor"],
        rating_diff_factor=elo_config["rating_diff_factor"],
        initial_rating=elo_config["initial_rating"],
    )
    winner_id, loser_id = 101, 102
    surface = "Clay"

    # --- FIX: Provide a surface ---
    assert calculator.get_player_rating(winner_id, surface) == 1500
    assert calculator.get_player_rating(loser_id, surface) == 1500

    # Simulate the match
    calculator.update_ratings(winner_id, loser_id, surface)

    # Winner change = 32 * (1 - 0.5) = +16
    # Loser change = 32 * (0 - 0.5) = -16
    assert calculator.get_player_rating(winner_id, surface) == 1516
    assert calculator.get_player_rating(loser_id, surface) == 1484


def test_elo_calculator_update_for_upset(elo_config):
    """
    Tests the rating update for an upset (a lower-rated player beats a higher-rated player).
    """
    calculator = EloCalculator(
        k_factor=elo_config["k_factor"],
        rating_diff_factor=elo_config["rating_diff_factor"],
        initial_rating=elo_config["initial_rating"],
    )

    winner_id, loser_id = 201, 202
    surface = "Grass"

    # --- FIX: Set ratings within the surface-specific dictionary ---
    calculator.elo_ratings[surface][winner_id] = 1400
    calculator.elo_ratings[surface][loser_id] = 1600

    # Simulate the upset
    # --- FIX: Provide a surface ---
    calculator.update_ratings(winner_id, loser_id, surface)

    # Expected change for a 200-point rating difference is ~24.3 points
    expected_change = 24.312
    assert calculator.get_player_rating(winner_id, surface) == pytest.approx(
        1400 + expected_change
    )
    assert calculator.get_player_rating(loser_id, surface) == pytest.approx(
        1600 - expected_change
    )
