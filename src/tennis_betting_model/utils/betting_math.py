import numpy as np
import pandas as pd


def add_ev_and_kelly(
    df: pd.DataFrame,
    prob_col: str = "predicted_prob",
    odds_col: str = "odds",
    commission: float = 0.0,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Adds Expected Value (EV) and Kelly Criterion columns to a DataFrame,
    with an option for commission-adjusted Kelly.
    """
    if not inplace:
        df = df.copy()

    # Expected Value remains the same (it's a pre-commission measure of value)
    df["expected_value"] = (df[prob_col] * (df[odds_col] - 1)) - (1 - df[prob_col])

    # Calculate Kelly Criterion with commission adjustment
    if commission > 0:
        # The formula for commission-adjusted Kelly is:
        # Kelly % = prob - ( (1 - prob) / ( (odds - 1) * (1 - commission) ) )
        # This correctly uses the net odds available after commission.
        kelly_denominator = (df[odds_col] - 1) * (1 - commission)
        df["kelly_fraction"] = df[prob_col] - ((1 - df[prob_col]) / kelly_denominator)
    else:
        # Original calculation if no commission is applied
        kelly_denominator = df[odds_col] - 1
        kelly_numerator = (df[prob_col] * kelly_denominator) - (1 - df[prob_col])
        df["kelly_fraction"] = kelly_numerator / kelly_denominator

    # Ensure Kelly is not negative (no bet) and handle cases where odds are 1.0 (denominator is zero)
    df["kelly_fraction"] = np.where(df[odds_col] > 1, df["kelly_fraction"], 0)
    df["kelly_fraction"] = df["kelly_fraction"].clip(lower=0)

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
