import numpy as np
import pandas as pd

from scripts.utils.logger import log_info
from scripts.utils.schema import enforce_schema, normalize_columns, patch_winner_column


def simulate_bankroll_growth(
    df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    strategy: str = "kelly",
    flat_stake_unit: float = 10.0,
) -> pd.DataFrame:
    """
    Simulates bankroll growth over a series of bets.
    """
    df = normalize_columns(df)
    df = patch_winner_column(df)
    if df.empty:
        log_info("DataFrame is empty, cannot run simulation.")
        return pd.DataFrame()

    df["bankroll"] = initial_bankroll
    df["strategy"] = strategy

    if strategy == "flat":
        df["stake"] = flat_stake_unit
    elif strategy == "kelly":
        df["stake"] = df["bankroll"] * df["kelly_fraction"]
    else:
        raise ValueError(f"Unknown staking strategy: {strategy}")

    df["profit"] = np.where(
        df["winner"] == 1, df["stake"] * (df["odds"] - 1), -df["stake"]
    )
    df["bankroll"] = df["bankroll"].iloc[0] + df["profit"].cumsum()
    return enforce_schema(df, "simulations")


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Simulate bankroll growth")
    parser.add_argument("--value_bets_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.value_bets_csv)
    simulation = simulate_bankroll_growth(df)
    simulation.to_csv(args.output_csv, index=False)
    log_info(f"Saved simulation to {args.output_csv}")


if __name__ == "__main__":
    main_cli()