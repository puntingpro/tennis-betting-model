import pandas as pd
import pytest

from scripts.utils.schema import enforce_schema, normalize_columns, patch_winner_column


def test_normalize_columns():
    df = pd.DataFrame(
        {
            "PlayerOne": ["A"],
            "PlayerTwo": ["B"],
            "WinnerName": ["A"],
            "someThing_else": [1],
        }
    )
    df_normalized = normalize_columns(df)
    assert "player_1" in df_normalized.columns
    assert "player_2" in df_normalized.columns
    assert "winner" in df_normalized.columns
    assert "something_else" in df_normalized.columns


def test_enforce_schema():
    df = pd.DataFrame({"market_id": [1], "ltp": [1.5]})
    df_enforced = enforce_schema(df, schema_name="matches")
    
    # Check that all required columns are present
    for col in ["selection_id", "volume", "timestamp", "match_id"]:
        assert col in df_enforced.columns
    
    # Check that the order is correct
    assert df_enforced.columns[0] == "market_id"
    assert df_enforced.columns[1] == "selection_id"


def test_patch_winner_column():
    df = pd.DataFrame({"winner": [1.0, 0.0, None, "1"]})
    df_patched = patch_winner_column(df)
    
    # Check that the dtype is integer and NaNs are filled
    assert df_patched["winner"].dtype == "int64"
    assert df_patched["winner"].tolist() == [1, 0, 0, 1]