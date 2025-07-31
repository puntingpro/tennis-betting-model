import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import cast, Optional
from .logger import log_info, log_error, log_success


class BetfairMatchLogSchema(pa.DataFrameModel):
    """Schema for the processed Betfair match log data."""

    match_id: Series[str] = pa.Field(nullable=False)
    tourney_date: Series[pa.DateTime] = pa.Field(nullable=False, coerce=True)
    tourney_name: Series[str] = pa.Field(nullable=True)
    winner_id: Series[int] = pa.Field(coerce=True)
    winner_historical_id: Series[float] = pa.Field(coerce=True, nullable=True)
    winner_name: Series[str] = pa.Field(nullable=True)
    loser_id: Series[int] = pa.Field(coerce=True)
    loser_historical_id: Series[float] = pa.Field(coerce=True, nullable=True)
    loser_name: Series[str] = pa.Field(nullable=True)
    # --- FIX: Add the new 'surface' column to the schema ---
    surface: Series[str] = pa.Field(isin=["Hard", "Clay", "Grass"])

    class Config:
        strict = True
        coerce = True


class FinalFeaturesSchema(pa.DataFrameModel):
    """
    Schema for the final feature-engineered DataFrame before model training.
    """

    # Match context
    match_id: Series[str] = pa.Field(nullable=False)
    tourney_date: Series[pa.DateTime] = pa.Field(nullable=False, coerce=True)
    tourney_name: Series[str] = pa.Field(nullable=True)
    surface: Series[str] = pa.Field(isin=["Hard", "Clay", "Grass"])

    # Player identifiers
    p1_id: Series[int] = pa.Field(coerce=True)
    p2_id: Series[int] = pa.Field(coerce=True)

    # Core features
    p1_rank: Series[float] = pa.Field(nullable=False, coerce=True)
    p2_rank: Series[float] = pa.Field(nullable=False, coerce=True)
    rank_diff: Series[float] = pa.Field(nullable=False, coerce=True)
    p1_elo: Series[float] = pa.Field(nullable=False, coerce=True)
    p2_elo: Series[float] = pa.Field(nullable=False, coerce=True)
    elo_diff: Series[float] = pa.Field(nullable=False, coerce=True)
    p1_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_matches_last_7_days: Series[int] = pa.Field(ge=0, coerce=True)
    p2_matches_last_7_days: Series[int] = pa.Field(ge=0, coerce=True)
    p1_matches_last_14_days: Series[int] = pa.Field(ge=0, coerce=True)
    p2_matches_last_14_days: Series[int] = pa.Field(ge=0, coerce=True)
    fatigue_diff_7_days: Series[int] = pa.Field(coerce=True)
    fatigue_diff_14_days: Series[int] = pa.Field(coerce=True)
    h2h_p1_wins: Series[int] = pa.Field(ge=0, coerce=True)
    h2h_p2_wins: Series[int] = pa.Field(ge=0, coerce=True)

    # Target variable
    winner: Series[int] = pa.Field(isin=[0, 1])

    p1_hand_R: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p1_hand_U: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p2_hand_R: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p2_hand_U: Optional[Series[int]] = pa.Field(isin=[0, 1])

    class Config:
        strict = False
        coerce = True


SCHEMA_REGISTRY = {
    "betfair_match_log": BetfairMatchLogSchema,
    "final_features": FinalFeaturesSchema,
}


def validate_data(df: pd.DataFrame, schema_name: str, context: str) -> pd.DataFrame:
    """
    Validates a DataFrame against a specified schema from the registry.
    """
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Schema '{schema_name}' not found in registry.")

    schema = SCHEMA_REGISTRY[schema_name]
    try:
        log_info(f"Validating schema for: {context}...")
        validated_df = schema.validate(df, lazy=True)
        log_success(f"✅ Schema validation successful for: {context}")
        return cast(pd.DataFrame, validated_df)
    except pa.errors.SchemaErrors as err:
        log_error(f"❌ Schema validation failed for: {context}")
        log_error("Validation errors:")
        failure_cases = err.failure_cases
        failure_cases["failure_case"] = failure_cases["failure_case"].astype(str)
        log_error(
            failure_cases.groupby(["column", "check"])["failure_case"]
            .first()
            .to_string()
        )
        raise
