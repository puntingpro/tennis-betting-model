# tests/modeling/test_train_eval_model.py

import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.scripts.modeling.train_eval_model import train_advanced_model


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Creates a small, balanced DataFrame for testing model training."""
    # --- MODIFIED: Expanded the dataset to prevent single-class splits ---
    data = {
        "p1_rank": [10, 20] * 10,
        "p2_rank": [20, 10] * 10,
        "rank_diff": [-10, 10] * 10,
        "p1_hand": ["R", "L"] * 10,
        "p2_hand": ["L", "R"] * 10,
        "winner": [1, 0] * 10,
        "match_id": range(20),
        "p1_id": [101, 102] * 10,
        "p2_id": [102, 101] * 10,
    }
    df = pd.DataFrame(data)
    required_cols = {
        "tourney_date": pd.to_datetime("2023-01-01"),
        "tourney_name": "Test Open",
        "surface": "Hard",
        "p1_elo": 1500,
        "p2_elo": 1500,
        "elo_diff": 0,
        "h2h_p1_wins": 0,
        "h2h_p2_wins": 0,
        "p1_win_perc": 0.5,
        "p2_win_perc": 0.5,
        "p1_surface_win_perc": 0.5,
        "p2_surface_win_perc": 0.5,
        "p1_form_last_10": 0.5,
        "p2_form_last_10": 0.5,
        "p1_height": 180,
        "p2_height": 180,
    }
    for col, val in required_cols.items():
        df[col] = val
    return df


def test_train_advanced_model(sample_feature_data):
    """
    Tests the advanced model training function to ensure it returns the
    correct types and that the metadata is structured as expected.
    """
    model, auc, meta = train_advanced_model(
        df=sample_feature_data,
        random_state=42,
        n_trials=2,  # Use a small number of trials for a quick test
    )

    assert isinstance(model, XGBClassifier)
    assert isinstance(auc, float) and 0.0 <= auc <= 1.0
    assert isinstance(meta, dict)
    assert meta["model_type"] == "XGBClassifier"
    assert "features" in meta
    assert "cross_val_auc" in meta