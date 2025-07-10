# tests/modeling/test_train_eval_model.py

import pandas as pd
import pytest

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
    for col in ['tourney_date', 'tourney_name', 'surface']:
        df[col] = ''
    return df


# --- MODIFIED: Update the test to use the correct function and parameters ---
def test_train_advanced_model(sample_feature_data):
    """
    Tests the advanced model training function.
    """
    model, report, auc, meta = train_advanced_model(
        df=sample_feature_data,
        test_size=0.5,
        random_state=42,
        n_trials=2  # Use a small number of trials for a quick test
    )
    
    assert model is not None
    assert isinstance(report, str)
    assert auc is not None and 0.0 <= auc <= 1.0
    assert meta["model_type"] == "XGBClassifier"
    assert meta["algorithm"] == "xgb"