# tests/pipeline/test_run_pipeline.py

import pandas as pd
import pytest
from unittest.mock import MagicMock

# Import the new, testable function directly
from src.scripts.pipeline.run_pipeline import process_markets

def test_process_markets_identifies_value_bet():
    """
    Tests that the core processing logic can take clean, mocked data
    and correctly identify a value bet by checking its return value.
    """
    # 1. --- Create perfect, clean mock data ---

    # Mock the ML model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.4, 0.6]] # Predicts 60% win prob for Player 1
    mock_model.feature_names_in_ = ['p1_rank', 'p2_rank', 'rank_diff', 'p1_height', 'p2_height', 'h2h_p1_wins', 'h2h_p2_wins', 'h2h_win_perc_p1', 'p1_win_perc', 'p2_win_perc', 'p1_surface_win_perc', 'p2_surface_win_perc', 'p1_hand_L', 'p1_hand_R', 'p1_hand_U', 'p2_hand_L', 'p2_hand_R', 'p2_hand_U']

    # Mock the market and book data from the API
    mock_market = MagicMock()
    mock_market.market_id = "1.2345"
    # --- FIX: Make market_start_time timezone-aware ---
    mock_market.market_start_time = pd.to_datetime("2023-10-26T12:00:00Z")
    mock_market.event.name = "Player A vs Player B"
    mock_market.competition.name = "Mock Open"
    mock_market.runners = [MagicMock(runner_name="Player A", selection_id=101), MagicMock(runner_name="Player B", selection_id=102)]
    
    mock_book = MagicMock()
    p1_runner = MagicMock()
    p1_runner.ex.available_to_back = [MagicMock(price=2.0)] # Odds of 2.0 for Player 1
    p2_runner = MagicMock()
    p2_runner.ex.available_to_back = [MagicMock(price=1.8)]
    mock_book.runners = [p1_runner, p2_runner]
    
    # Mock the lookup dictionaries and DataFrames
    mock_player_info_lookup = {
        101: {'hand': 'R', 'height': 180.0},
        102: {'hand': 'L', 'height': 185.0}
    }
    mock_df_rankings = pd.DataFrame({
        # --- FIX: Make ranking_date timezone-aware ---
        'ranking_date': pd.to_datetime(['2023-01-01', '2023-01-01'], utc=True),
        'player': [101, 102],
        'rank': [10.0, 25.0]
    }).sort_values(by='ranking_date')
    
    mock_betting_config = {'ev_threshold': 0.1}

    # 2. --- Run the core logic function directly ---
    result = process_markets(
        model=mock_model,
        market_catalogues=[mock_market],
        market_book_lookup={"1.2345": mock_book},
        player_info_lookup=mock_player_info_lookup,
        df_rankings=mock_df_rankings,
        betting_config=mock_betting_config
    )

    # 3. --- Assert the results ---
    assert len(result) == 1
    
    value_bet = result[0]
    assert value_bet['player_name'] == "Player A"
    assert value_bet['odds'] == 2.0
    assert value_bet['EV'] == "+20.00%"