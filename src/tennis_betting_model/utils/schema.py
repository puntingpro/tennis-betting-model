import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from typing import cast, Optional
from .logger import log_info, log_error, log_success
from pathlib import Path


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
    # --- FIX: Allow 'Unknown' as a valid surface ---
    surface: Series[str] = pa.Field(isin=["Hard", "Clay", "Grass", "Unknown"])

    class Config:
        strict = True
        coerce = True


# ... (rest of the file is unchanged) ...
class FinalFeaturesSchema(pa.DataFrameModel):
    """
    Schema for the final feature-engineered DataFrame before model training.
    """

    market_id: Series[str] = pa.Field(nullable=False)
    tourney_date: Series[pa.DateTime] = pa.Field(nullable=False, coerce=True)
    tourney_name: Series[str] = pa.Field(nullable=True)
    # --- FIX: Allow 'Unknown' as a valid surface ---
    surface: Series[str] = pa.Field(isin=["Hard", "Clay", "Grass", "Unknown"])
    p1_id: Series[int] = pa.Field(coerce=True)
    p2_id: Series[int] = pa.Field(coerce=True)
    p1_rank: Series[float] = pa.Field(nullable=False, coerce=True)
    p2_rank: Series[float] = pa.Field(nullable=False, coerce=True)
    rank_diff: Series[float] = pa.Field(nullable=False, coerce=True)
    p1_elo: Series[float] = pa.Field(nullable=True, coerce=True)
    p2_elo: Series[float] = pa.Field(nullable=True, coerce=True)
    elo_diff: Series[float] = pa.Field(nullable=True, coerce=True)
    p1_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_surface_win_perc: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_form: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p2_form: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    p1_matches_last_7_days: Series[int] = pa.Field(ge=0, coerce=True)
    p2_matches_last_7_days: Series[int] = pa.Field(ge=0, coerce=True)
    p1_matches_last_14_days: Series[int] = pa.Field(ge=0, coerce=True)
    p2_matches_last_14_days: Series[int] = pa.Field(ge=0, coerce=True)
    fatigue_diff_7_days: Series[int] = pa.Field(coerce=True)
    fatigue_diff_14_days: Series[int] = pa.Field(coerce=True)
    h2h_p1_wins: Series[int] = pa.Field(ge=0, coerce=True)
    h2h_p2_wins: Series[int] = pa.Field(ge=0, coerce=True)
    winner: Series[int] = pa.Field(isin=[0, 1])
    p1_hand_R: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p1_hand_U: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p2_hand_R: Optional[Series[int]] = pa.Field(isin=[0, 1])
    p2_hand_U: Optional[Series[int]] = pa.Field(isin=[0, 1])

    class Config:
        strict = False
        coerce = True


class ConsolidatedRankingsSchema(pa.DataFrameModel):
    """Schema for the consolidated historical rankings data."""

    ranking_date: Series[pa.DateTime] = pa.Field(nullable=False, coerce=True)
    rank: Series[int] = pa.Field(nullable=False, coerce=True)
    player: Series[int] = pa.Field(nullable=False, coerce=True)
    points: Optional[Series[str]] = pa.Field(nullable=True)
    tours: Optional[Series[str]] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class RawPlayersSchema(pa.DataFrameModel):
    """Schema for the consolidated player attributes file."""

    player_id: Series[int] = pa.Field(coerce=True)
    first_name: Series[str] = pa.Field(nullable=True)
    last_name: Series[str] = pa.Field(nullable=True)
    hand: Series[str] = pa.Field(nullable=True)
    dob: Series[str] = pa.Field(nullable=True)
    country_ioc: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class PlayerMapSchema(pa.DataFrameModel):
    """Schema for the generated player mapping file."""

    betfair_id: Series[int] = pa.Field(coerce=True)
    historical_id: Series[float] = pa.Field(coerce=True, nullable=True)
    betfair_name: Series[str]
    matched_name: Series[str]
    confidence: Series[float] = pa.Field(ge=0, le=100, coerce=True)
    method: Series[str]

    class Config:
        strict = True
        coerce = True


SCHEMA_REGISTRY = {
    "betfair_match_log": BetfairMatchLogSchema,
    "final_features": FinalFeaturesSchema,
    "consolidated_rankings": ConsolidatedRankingsSchema,
    "raw_players": RawPlayersSchema,
    "player_map": PlayerMapSchema,
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

        failure_cases = err.failure_cases
        failure_cases["failure_case"] = failure_cases["failure_case"].astype(str)

        log_error("Validation error summary:")
        log_error(
            failure_cases.groupby(["column", "check"])["failure_case"]
            .first()
            .to_string()
        )

        failure_log_path = Path("data/analysis/validation_failures.csv")
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        failure_cases.to_csv(failure_log_path, index=False)
        log_error(f"Full failure report saved to: {failure_log_path}")

        raise
