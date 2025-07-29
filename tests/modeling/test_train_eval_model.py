# tests/modeling/test_train_eval_model.py

import pandas as pd
import pytest
from xgboost import XGBClassifier
import numpy as np
import joblib
import optuna

from tennis_betting_model.modeling.train_eval_model import train_eval_model


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """
    Creates a more robust, balanced DataFrame for testing model training.
    The relationship between rank_diff and the winner is intentionally made non-random.
    """
    data = {
        "p1_rank": [10, 20, 5, 50, 15, 80, 5, 90, 12, 45] * 4,
        "p2_rank": [20, 10, 50, 5, 80, 15, 90, 5, 45, 12] * 4,
        "p1_hand": ["R", "L", "R", "R", "L", "R", "R", "L", "R", "L"] * 4,
        "p2_hand": ["L", "R", "R", "L", "R", "R", "L", "R", "L", "R"] * 4,
    }
    df = pd.DataFrame(data)

    # Create a predictable winner based on rank difference
    df["rank_diff"] = df["p1_rank"] - df["p2_rank"]
    df["winner"] = np.where(
        df["rank_diff"] < 0, 1, 0
    )  # Lower rank (p1) is more likely to win

    # Add other required columns that are not the focus of this test
    df["match_id"] = range(len(df))
    df["p1_id"] = [101, 102] * (len(df) // 2)
    df["p2_id"] = [102, 101] * (len(df) // 2)
    df["tourney_date"] = pd.to_datetime("2023-01-01")
    df["tourney_name"] = "Test Open"
    df["surface"] = "Hard"

    return df


def test_train_eval_model_outputs_correct_types(tmp_path, sample_feature_data):
    """
    Tests that the model training function returns the correct object types and
    that the model file is created.
    """
    model_path = tmp_path / "test_model.joblib"

    train_eval_model(
        data=sample_feature_data,
        model_output_path=str(model_path),
        n_trials=2,  # Quick test with minimal trials
    )

    assert model_path.exists()
    model = joblib.load(model_path)
    assert isinstance(model, XGBClassifier)


def test_train_eval_model_logic_is_sound(tmp_path, sample_feature_data):
    """
    Tests the model training logic to ensure it produces a sensible model
    with plausible performance on a predictable dataset.
    """
    model_path = tmp_path / "test_model_logic.joblib"

    # Suppress Optuna's trial logging for cleaner test output
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    train_eval_model(
        data=sample_feature_data,
        model_output_path=str(model_path),
        n_trials=5,
        test_size=0.5,  # Use a larger test size for stability with small data
    )

    model = joblib.load(model_path)

    # 1. Check that feature importances are reasonable
    # For this dataset, rank_diff should be the most important feature.
    feature_importances = model.get_booster().get_score(importance_type="weight")
    assert "rank_diff" in feature_importances
    # Ensure rank_diff is the most important or one of the most important features
    assert feature_importances["rank_diff"] == max(feature_importances.values())

    # 2. Check that the feature names in the model match the input columns (after one-hot encoding)
    expected_features = set(
        ["p1_rank", "p2_rank", "rank_diff", "p1_hand_R", "p2_hand_R"]
    )
    model_features = set(model.feature_names_in_)
    assert expected_features.issubset(model_features)
