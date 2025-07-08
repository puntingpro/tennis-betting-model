# src/scripts/utils/schema.py

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

class ValueBetsSchema(pa.SchemaModel):
    match_id: Series[str] = pa.Field(nullable=False)
    player_1: Series[str] = pa.Field(nullable=True)
    player_2: Series[str] = pa.Field(nullable=True)
    predicted_prob: Series[float] = pa.Field(ge=0, le=1)
    odds: Series[float] = pa.Field(gt=1)
    expected_value: Series[float] = pa.Field()
    kelly_fraction: Series[float] = pa.Field()
    confidence_score: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        strict = True
        coerce = True

def validate_data(df: pd.DataFrame, schema: pa.SchemaModel) -> DataFrame:
    """Validates a DataFrame against a pandera schema."""
    try:
        schema.validate(df, lazy=True)
        return df
    except pa.errors.SchemaErrors as err:
        print("Schema validation errors:")
        print(err.failure_cases)
        raise

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names to lowercase and replaces spaces with underscores.
    """
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def patch_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the 'winner' column, ensuring it's numeric.
    It checks for a 'result' column and converts it if 'winner' is missing.
    """
    if 'winner' not in df.columns and 'result' in df.columns:
        df['winner'] = pd.to_numeric(df['result'], errors='coerce')
    elif 'winner' in df.columns:
        df['winner'] = pd.to_numeric(df['winner'], errors='coerce')
        
    return df