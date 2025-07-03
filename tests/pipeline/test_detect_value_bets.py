import pandas as pd
import pytest

from scripts.pipeline.detect_value_bets import detect_value_bets


def test_detect_value_bets_integration():
    # Sample data that includes columns for odds, predicted_prob, and winner
    data = {
        "match_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
        "player_1": ["A", "B", "C", "D", "E", "F"],
        "player_2": ["B", "A", "D", "C", "F", "E"],
        "odds": [2.0, 2.0, 1.5, 3.0, 5.0, 1.25],
        "predicted_prob": [0.6, 0.4, 0.7, 0.3, 0.3, 0.7],
        "winner": [1, 0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    # Detect value bets with a threshold of 10%
    value_bets = detect_value_bets(df, ev_threshold=0.1)

    # Expected value for player A: (0.6 * (2.0 - 1)) - (1 - 0.6) = 0.2
    # Expected value for player C: (0.7 * (1.5 - 1)) - (1 - 0.7) = 0.05 (not a value bet)
    assert len(value_bets) == 1
    assert value_bets.iloc[0]["player_1"] == "A"
    assert pytest.approx(value_bets.iloc[0]["expected_value"]) == 0.2