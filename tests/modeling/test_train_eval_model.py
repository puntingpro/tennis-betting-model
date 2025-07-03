import pandas as pd
import pytest

from scripts.modeling.train_eval_model import run_train_eval_model

@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Creates a small, balanced DataFrame for testing model training."""
    data = {
        "match_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "player_1": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "player_2": ["B", "A", "D", "C", "F", "E", "H", "G"],
        "implied_prob_1": [0.5, 0.5, 0.6, 0.4, 0.2, 0.8, 0.9, 0.1],
        "implied_prob_2": [0.5, 0.5, 0.4, 0.6, 0.8, 0.2, 0.1, 0.9],
        "implied_prob_diff": [0.0, 0.0, 0.2, -0.2, -0.6, 0.6, 0.8, -0.8],
        "odds_margin": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "winner": [1, 0, 0, 1, 0, 1, 1, 0],
    }
    return pd.DataFrame(data)


def test_run_train_eval_model_with_xgb(sample_feature_data):
    # Test with XGBoost algorithm
    model, report, auc, meta = run_train_eval_model(
        sample_feature_data, algorithm="xgb"
    )
    assert model is not None
    assert isinstance(report, str)
    assert auc is not None and 0.0 <= auc <= 1.0
    assert meta["algorithm"] == "xgb"