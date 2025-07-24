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
