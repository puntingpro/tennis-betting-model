import pandas as pd

from scripts.utils.betting_math import add_ev_and_kelly
from scripts.utils.logger import log_info
from scripts.utils.schema import enforce_schema, normalize_columns


def validate_value_bets(df: pd.DataFrame) -> None:
    """
    Validates that essential columns have no missing values.
    """
    key_cols = ["match_id", "player_1", "player_2", "odds", "predicted_prob"]
    missing = df[key_cols].isnull().sum()
    if missing.any():
        raise ValueError(f"Validation failed: Found missing values in key columns:\n{missing[missing > 0]}")


def detect_value_bets(
    df: pd.DataFrame,
    ev_threshold: float = 0.1,
    confidence_threshold: float = 0.5,
    max_odds: float = 10.0,
    max_margin: float = 1.05,
) -> pd.DataFrame:
    """
    Filters a DataFrame of predictions to find value bets.
    """
    df = normalize_columns(df)
    validate_value_bets(df)
    
    df = add_ev_and_kelly(df, inplace=False)

    value_bets_df = df[
        (df["expected_value"] > ev_threshold)
        & (df["predicted_prob"] > confidence_threshold)
        & (df["odds"] < max_odds)
    ].copy()

    # Add a confidence score for analysis
    value_bets_df["confidence_score"] = (
        value_bets_df["predicted_prob"] - (1 / value_bets_df["odds"])
    )

    log_info(f"Found {len(value_bets_df)} value bets.")
    return enforce_schema(value_bets_df, "value_bets")


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Detect value bets from predictions")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    value_bets = detect_value_bets(df)
    value_bets.to_csv(args.output_csv, index=False)
    log_info(f"Saved {len(value_bets)} value bets to {args.output_csv}")


if __name__ == "__main__":
    main_cli()