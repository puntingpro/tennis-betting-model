# tests/modeling/test_train_eval_model.py

import pandas as pd
import pytest
from xgboost import XGBClassifier

# --- MODIFIED: Import the correct function ---
from src.scripts.modeling.train_eval_model import train_advanced_model

@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Creates a small, balanced DataFrame for testing model training."""
    # Using a more realistic feature set that the model expects
    data = {
        "p1_rank": [10, 20, 30, 40, 5, 15, 25, 35],
        "p2_rank": [20, 10, 40, 30, 15, 5, 35, 25],
        "rank_diff": [-10, 10, -10, 10, -10, 10, -10, 10],
        "p1_hand": ['R', 'L', 'R', 'R', 'L', 'R', 'L', 'L'],
        "p2_hand": ['L', 'R', 'R', 'L', 'R', 'L', 'L', 'R'],
        "winner": [1, 0, 0, 1, 1, 0, 1, 0],
        "match_id": [1, 1, 2, 2, 3, 3, 4, 4], # Needed for balancing logic
        "p1_id": [101, 102, 103, 104, 105, 106, 107, 108], # Dummy IDs
        "p2_id": [102, 101, 104, 103, 106, 105, 108, 107]  # Dummy IDs
    }
    # Add other columns expected by the training function with default values
    df = pd.DataFrame(data)
    # Add all other required columns with default values to satisfy schema validation
    required_cols = {
        'tourney_date': pd.to_datetime('2023-01-01'), 'tourney_name': 'Test Open', 'surface': 'Hard',
        'p1_elo': 1500, 'p2_elo': 1500, 'elo_diff': 0, 'h2h_p1_wins': 0, 'h2h_p2_wins': 0,
        'p1_win_perc': 0.5, 'p2_win_perc': 0.5, 'p1_surface_win_perc': 0.5, 'p2_surface_win_perc': 0.5,
        'p1_form_last_10': 0.5, 'p2_form_last_10': 0.5, 'p1_height': 180, 'p2_height': 180
    }
    for col, val in required_cols.items():
        df[col] = val
        
    return df


# --- MODIFIED: Update the test to match the new return signature ---
def test_train_advanced_model(sample_feature_data):
    """
    Tests the advanced model training function to ensure it returns the
    correct types and that the metadata is structured as expected.
    """
    model, auc, meta = train_advanced_model(
        df=sample_feature_data,
        random_state=42,
        n_trials=2  # Use a small number of trials for a quick test
    )
    
    assert isinstance(model, XGBClassifier)
    assert isinstance(auc, float) and 0.0 <= auc <= 1.0
    assert isinstance(meta, dict)
    assert meta["model_type"] == "XGBClassifier"
    assert "features" in meta
    assert "cross_val_auc" in meta