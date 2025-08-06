# src/scripts/pipeline/simulate_bankroll_growth.py

import pandas as pd
from typing import Dict

from tennis_betting_model.utils.logger import log_info, log_warning
from tennis_betting_model.utils.common import (
    normalize_df_column_names,
    patch_winner_column,
)


def calculate_max_drawdown(bankroll_series: pd.Series) -> tuple[float, float]:
    """Calculates the maximum drawdown and the peak bankroll."""
    peak = bankroll_series.expanding(min_periods=1).max()
    drawdown = (bankroll_series - peak) / peak
    max_drawdown = drawdown.min()
    peak_bankroll = peak.max()
    return peak_bankroll, max_drawdown if pd.notna(max_drawdown) else 0.0


def simulate_bankroll_growth(
    df: pd.DataFrame,
    simulation_params: Dict,
    initial_bankroll: float,
    strategy: str = "kelly",
    stake_unit: float = 10.0,
    kelly_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Simulates bankroll growth over a series of bets with multiple strategies.
    """
    max_kelly_stake_fraction = simulation_params.get("max_kelly_stake_fraction", 0.1)
    max_profit_per_bet = simulation_params.get("max_profit_per_bet", 10000.0)

    df = normalize_df_column_names(df)
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
            row_kelly_fraction = float(row.get("kelly_fraction", 0.0))
            row_odds = float(row.get("odds", 1.0))
            row_winner = int(row.get("winner", 0))

            if strategy == "kelly":
                kelly_frac = row_kelly_fraction * float(kelly_fraction)
                kelly_frac = min(kelly_frac, max_kelly_stake_fraction)
                current_stake = bankroll * kelly_frac
            elif strategy == "flat":
                current_stake = float(stake_unit)
            elif strategy == "percent":
                current_stake = bankroll * (float(stake_unit) / 100.0)

            current_stake = max(0.0, min(current_stake, bankroll))

            if row_winner == 1:
                profit = current_stake * (row_odds - 1.0)
                profit = min(profit, max_profit_per_bet)
            else:
                profit = -current_stake

        except (ValueError, TypeError) as e:
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
