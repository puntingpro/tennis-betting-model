# src/scripts/utils/schema.py

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

# --- MODIFIED: Use the more compatible pa.DataFrameModel ---
class RawMatchesSchema(pa.DataFrameModel):
    """
    Schema for the raw consolidated match data before feature engineering.
    Ensures that essential columns are present and have the correct data type.
    """
    tourney_date: Series[pd.Timestamp] = pa.Field(nullable=False)
    tourney_name: Series[str] = pa.Field(nullable=True)
    surface: Series[str] = pa.Field(nullable=True)
    match_num: Series[int] = pa.Field(coerce=True)
    winner_id: Series[int] = pa.Field(coerce=True)
    loser_id: Series[int] = pa.Field(coerce=True)

    class Config:
        strict = False
        coerce = True

class PlayerFeaturesSchema(pa.DataFrameModel):
    """
    Schema for the final feature-engineered DataFrame.
    Validates the data just before it's used for model training or backtesting.
    """
    match_id: Series[str] = pa.Field(nullable=False)
    tourney_date: Series[pd.Timestamp] = pa.Field(nullable=False)
    surface: Series[str] = pa.Field(nullable=True)
    p1_id: Series[int] = pa.Field(coerce=True)
    p2_id: Series[int] = pa.Field(coerce=True)
    winner: Series[int] = pa.Field(isin=[0, 1])

    # Key Features
    p1_rank: Series[float] = pa.Field(nullable=True, coerce=True)
    p2_rank: Series[float] = pa.Field(nullable=True, coerce=True)
    rank_diff: Series[float] = pa.Field(nullable=True, coerce=True)
    h2h_p1_wins: Series[int] = pa.Field(ge=0, coerce=True)
    h2h_p2_wins: Series[int] = pa.Field(ge=0, coerce=True)
    p1_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    class Config:
        strict = False
        coerce = True

class BacktestResultsSchema(pa.DataFrameModel):
    """
    Schema for the output of the backtest_strategy.py script.
    Ensures the results have the expected columns and data types before analysis.
    """
    match_id: Series[str] = pa.Field(nullable=False)
    tourney_name: Series[str] = pa.Field(nullable=True)
    odds: Series[float] = pa.Field(gt=1)
    predicted_prob: Series[float] = pa.Field(ge=0, le=1)
    winner: Series[int] = pa.Field(isin=[0, 1])
    expected_value: Series[float] = pa.Field()
    kelly_fraction: Series[float] = pa.Field()

    class Config:
        strict = True
        coerce = True

def validate_data(df: pd.DataFrame, schema: pa.DataFrameModel, context: str) -> DataFrame:
    """
    Validates a DataFrame against a pandera schema, providing a clear context on error.
    """
    try:
        print(f"Validating schema for: {context}...")
        validated_df = schema.validate(df, lazy=True)
        print(f"✅ Schema validation successful for: {context}")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print(f"❌ Schema validation failed for: {context}")
        print("Failure cases:")
        print(err.failure_cases)
        raise