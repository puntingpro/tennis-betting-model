import pandas as pd
from types import SimpleNamespace
from decimal import Decimal, getcontext

# Add project root to the Python path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the function we want to verify
from src.scripts.pipeline.run_pipeline import process_markets
from src.scripts.utils.logger import setup_logging

def run_verification():
    """
    Directly verifies the core logic of process_markets outside of the
    pytest framework to confirm its correctness.
    """
    print("--- Starting Logic Verification ---")
    setup_logging() # Setup logging to see potential errors from the function

    # 1. --- Create simple, clean data objects ---

    # Mock the ML model's behavior
    model = SimpleNamespace(
        predict_proba=lambda features: [[0.4, 0.6]], # 60% win prob for P1
        feature_names_in_=['p1_rank', 'p2_rank', 'rank_diff', 'p1_height', 'p2_height', 'h2h_p1_wins', 'h2h_p2_wins', 'h2h_win_perc_p1', 'p1_win_perc', 'p2_win_perc', 'p1_surface_win_perc', 'p2_surface_win_perc', 'p1_hand_L', 'p1_hand_R', 'p1_hand_U', 'p2_hand_L', 'p2_hand_R', 'p2_hand_U']
    )

    # Mock the market and book data from the API
    p1_runner_data = SimpleNamespace(price=2.0)
    p2_runner_data = SimpleNamespace(price=1.8)
    
    market_runner1 = SimpleNamespace(runner_name="Player A", selection_id=101)
    market_runner2 = SimpleNamespace(runner_name="Player B", selection_id=102)
    
    book_runner1 = SimpleNamespace(ex=SimpleNamespace(available_to_back=[p1_runner_data]))
    book_runner2 = SimpleNamespace(ex=SimpleNamespace(available_to_back=[p2_runner_data]))

    market = SimpleNamespace(
        market_id="1.2345",
        market_start_time=pd.to_datetime("2023-10-26T12:00:00Z"),
        event=SimpleNamespace(name="Player A vs Player B"),
        competition=SimpleNamespace(name="Mock Open"),
        runners=[market_runner1, market_runner2]
    )
    
    book = SimpleNamespace(runners=[book_runner1, book_runner2])
    
    # Create the lookup dictionaries and DataFrames
    player_info_lookup = {
        101: {'hand': 'R', 'height': 180.0},
        102: {'hand': 'L', 'height': 185.0}
    }
    df_rankings = pd.DataFrame({
        'ranking_date': pd.to_datetime(['2023-01-01', '2023-01-01']),
        'player': [101, 102],
        'rank': [10.0, 25.0]
    }).sort_values(by='ranking_date')
    
    betting_config = {'ev_threshold': 0.1}

    # 2. --- Run the core logic function directly ---
    print("\nCalling process_markets function...")
    result = process_markets(
        model=model,
        market_catalogues=[market],
        market_book_lookup={"1.2345": book},
        player_info_lookup=player_info_lookup,
        df_rankings=df_rankings,
        betting_config=betting_config
    )

    # 3. --- Assert and print the results ---
    print("\n--- Verification Results ---")
    print(f"Function returned: {result}")
    
    # EV = (0.6 * 2.0) - 1 = 0.2
    # Since 0.2 > 0.1 threshold, a bet should be found.
    assert len(result) == 1
    
    value_bet = result[0]
    assert value_bet['player_name'] == "Player A"
    assert value_bet['odds'] == 2.0
    assert value_bet['EV'] == "+20.00%"
    
    print("\n✅ SUCCESS: The core logic is correct!")
    print("--- Verification Complete ---")


if __name__ == "__main__":
    run_verification()