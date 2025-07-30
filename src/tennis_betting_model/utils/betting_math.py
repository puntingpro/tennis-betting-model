import numpy as np
import pandas as pd


def add_ev_and_kelly(
    df: pd.DataFrame,
    prob_col: str = "predicted_prob",
    odds_col: str = "odds",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Adds Expected Value (EV) and Kelly Criterion columns to a DataFrame.
    """
    if not inplace:
        df = df.copy()

    # Calculate Expected Value
    df["expected_value"] = (df[prob_col] * (df[odds_col] - 1)) - (1 - df[prob_col])

    # Calculate Kelly Criterion
    # Kelly fraction = (p * (b - 1) - (1 - p)) / (b - 1)
    # where p = probability of winning, b = decimal odds
    kelly_numerator = (df[prob_col] * (df[odds_col] - 1)) - (1 - df[prob_col])
    kelly_denominator = df[odds_col] - 1
    # Avoid division by zero for odds of 1.0
    df["kelly_fraction"] = np.where(
        kelly_denominator > 0, kelly_numerator / kelly_denominator, 0
    )

    return df


def calculate_pnl(df: pd.DataFrame, commission: float = 0.05) -> pd.DataFrame:
    """
    Ensures a 'pnl' column exists, calculating it if necessary.
    """
    if "pnl" in df.columns and not df["pnl"].isnull().all():
        return df

    df["pnl"] = df.apply(
        lambda row: (row["odds"] - 1) * (1 - commission) if row["winner"] == 1 else -1,
        axis=1,
    )
    return df
