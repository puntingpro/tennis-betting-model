import pandas as pd
import pytest
from unittest.mock import MagicMock

from tennis_betting_model.pipeline.value_finder import MarketProcessor
from tennis_betting_model.builders.feature_builder import FeatureBuilder


@pytest.fixture
def mock_dependencies():
    """Mocks all dependencies needed for the MarketProcessor."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.4, 0.6]]  # P1 has 60% chance
    mock_model.feature_names_in_ = [
        "p1_rank",
        "p2_rank",
        "rank_diff",
        "p1_hand_R",
        "p2_hand_R",
    ]

    mock_feature_builder = MagicMock(spec=FeatureBuilder)
    mock_feature_builder.build_features.return_value = {
        "p1_rank": 10,
        "p2_rank": 20,
        "rank_diff": -10,
        "p1_hand_R": 1,
        "p2_hand_R": 0,
    }

    mock_betting_config = {"ev_threshold": 0.1}

    return mock_model, mock_feature_builder, mock_betting_config


@pytest.fixture
def mock_market_data():
    """Provides mock market catalogue and book data."""
    mock_market = MagicMock()
    mock_market.market_id = "1.123"
    mock_market.market_name = "ATP Challenger Hard"
    mock_market.market_start_time = pd.to_datetime("2023-10-26T12:00:00Z")
    mock_market.competition.name = "Test Open"
    mock_market.event.name = "Player A vs Player B"
    mock_market.runners = [
        MagicMock(runner_name="Player A", selection_id=101),
        MagicMock(runner_name="Player B", selection_id=102),
    ]

    mock_book = MagicMock()
    p1_runner_book = MagicMock()
    p1_runner_book.selection_id = 101
    # FIX: Use a list of dicts to correctly mock the price data structure
    p1_runner_book.ex.available_to_back = [{"price": 2.0, "size": 100}]  # Value bet
    p2_runner_book = MagicMock()
    p2_runner_book.selection_id = 102
    # FIX: Use a list of dicts to correctly mock the price data structure
    p2_runner_book.ex.available_to_back = [{"price": 1.8, "size": 100}]  # Not value
    mock_book.runners = [p1_runner_book, p2_runner_book]

    return mock_market, mock_book


def test_market_processor_identifies_value(mock_dependencies, mock_market_data):
    """
    Tests that the MarketProcessor correctly identifies a value bet when
    the EV is above the threshold.
    """
    model, feature_builder, config = mock_dependencies
    market_cat, market_book = mock_market_data

    processor = MarketProcessor(model, feature_builder, config)
    result = processor.process_market(market_cat, market_book)

    assert len(result) == 1
    bet = result[0]
    assert bet["player_name"] == "Player A"
    assert bet["odds"] == 2.0
    assert bet["model_prob"] == "60.00%"
    assert bet["ev"] == "+20.00%"
    assert bet["selection_id"] == 101


def test_market_processor_ignores_no_value(mock_dependencies, mock_market_data):
    """
    Tests that the MarketProcessor returns no bets when the EV is below
    the threshold.
    """
    model, feature_builder, config = mock_dependencies
    market_cat, market_book = mock_market_data

    # Modify model prediction so there is no value
    model.predict_proba.return_value = [[0.6, 0.4]]  # P1 has 40% chance

    processor = MarketProcessor(model, feature_builder, config)
    result = processor.process_market(market_cat, market_book)

    assert len(result) == 0
