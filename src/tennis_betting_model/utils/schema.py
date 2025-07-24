# src/scripts/utils/schema.py

import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import cast


class PlayerFeaturesSchema(pa.DataFrameModel):
    """
    Schema for the final feature-engineered DataFrame.
    Validates the data just before it's used for model training or backtesting.
    """

    match_id: Series[str] = pa.Field(nullable=False)
    tourney_date: Series[pa.DateTime] = pa.Field(nullable=False)  # pyright: ignore
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


SCHEMA_REGISTRY = {
    "model_training_input": PlayerFeaturesSchema,
    "player_features": PlayerFeaturesSchema,
}


def validate_data(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Validates a DataFrame against a specified schema from the registry.

    Args:
        df: The DataFrame to validate.
        schema_name: The name of the schema to use for validation.

    Returns:
        The validated DataFrame.

    Raises:
        ValueError: If the schema name is not found in the registry.
        pa.errors.SchemaErrors: If the DataFrame fails validation.
    """
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Schema '{schema_name}' not found in registry.")

    schema = SCHEMA_REGISTRY[schema_name]
    try:
        print(f"Validating schema for: {schema_name}...")
        validated_df = schema.validate(df, lazy=True)
        print(f"✅ Schema validation successful for: {schema_name}")
        return cast(pd.DataFrame, validated_df)
    except pa.errors.SchemaErrors as err:
        print(f"❌ Schema validation failed for: {schema_name}")
        print("Failure cases:")
        print(err.failure_cases)
        raise
