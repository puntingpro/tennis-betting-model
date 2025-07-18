# src/scripts/pipeline/simulate_bankroll_growth.py

import numpy as np
import pandas as pd
from pathlib import Path

from src.scripts.utils.logger import log_info, log_warning
from src.scripts.utils.common import normalize_columns, patch_winner_column
from src.scripts.utils.constants import DEFAULT_INITIAL_BANKROLL

MAX_KELLY_STAKE_FRACTION = 0.1  # Cap stakes at 10% of the bankroll
MAX_PROFIT_PER_BET = 10000.0  # Cap profit on any single bet to $10,000


def calculate_max_drawdown(bankroll_series: pd.Series) -> tuple[float, float]:
    """Calculates the maximum drawdown and the peak bankroll."""
    peak = bankroll_series.expanding(min_periods=1).max()
    drawdown = (bankroll_series - peak) / peak
    max_drawdown = drawdown.min()
    peak_bankroll = peak.max()
    return peak_bankroll, max_drawdown if pd.notna(max_drawdown) else 0.0


def simulate_bankroll_growth(
    df: pd.DataFrame,
    initial_bankroll: float,
    strategy: str = "kelly",
    stake_unit: float = 10.0,
    kelly_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Simulates bankroll growth over a series of bets with multiple strategies.
    """
    df = normalize_columns(df)
    df = patch_winner_column(df)
    if df.empty:
        log_info("DataFrame is empty, cannot run simulation.")
        return pd.DataFrame()

    if "tourney_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["tourney_date"]
    ):
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    df.dropna(subset=["tourney_date"], inplace=True)
    df = df.sort_values(by="tourney_date").reset_index(drop=True)

    bankroll = float(initial_bankroll)
    stakes = []
    profits = []
    bankroll_history = []

    for _, row in df.iterrows():
        profit = 0.0
        current_stake = 0.0

        try:
            # --- BUG FIX: Add robust casting to prevent calculation errors ---
            row_kelly_fraction = float(row.get("kelly_fraction", 0.0))
            row_odds = float(row.get("odds", 1.0))
            row_winner = int(row.get("winner", 0))
            # --- END FIX ---

            if strategy == "kelly":
                kelly_frac = row_kelly_fraction * float(kelly_fraction)
                kelly_frac = min(kelly_frac, MAX_KELLY_STAKE_FRACTION)
                current_stake = bankroll * kelly_frac
            elif strategy == "flat":
                current_stake = float(stake_unit)
            elif strategy == "percent":
                current_stake = bankroll * (float(stake_unit) / 100.0)

            current_stake = max(0.0, min(current_stake, bankroll))

            if row_winner == 1:
                profit = current_stake * (row_odds - 1.0)
                profit = min(profit, MAX_PROFIT_PER_BET)
            else:
                profit = -current_stake

        except (ValueError, TypeError) as e:
            # If any row has bad data, log it, skip the bet, and continue
            log_warning(
                f"Skipping a row in simulation due to data error: {e}. Row: {row.to_dict()}"
            )
            profit = 0.0
            current_stake = 0.0

        bankroll += profit

        stakes.append(current_stake)
        profits.append(profit)
        bankroll_history.append(bankroll)

    df["stake"] = stakes
    df["profit"] = profits
    df["bankroll"] = bankroll_history

    return df
