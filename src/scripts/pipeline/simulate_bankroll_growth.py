# src/scripts/pipeline/simulate_bankroll_growth.py

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from src.scripts.utils.logger import log_info, log_success, setup_logging
from src.scripts.utils.common import normalize_columns, patch_winner_column
from src.scripts.utils.constants import DEFAULT_INITIAL_BANKROLL


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

    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
        df = df.sort_values(by="tourney_date").reset_index(drop=True)

    df["bankroll_start"] = initial_bankroll
    df["strategy"] = strategy

    # Calculate stake based on the chosen strategy
    if strategy == "flat":
        df["stake"] = stake_unit
    elif strategy == "percent":
        df["stake"] = df["bankroll_start"].shift(1).fillna(initial_bankroll) * (
            stake_unit / 100
        )
    elif strategy == "kelly":
        df["kelly_fraction"] = pd.to_numeric(
            df["kelly_fraction"], errors="coerce"
        ).fillna(0)
        # Apply the fraction limit (e.g., half-Kelly)
        limited_kelly = df["kelly_fraction"] * kelly_fraction
        df["stake"] = (
            df["bankroll_start"].shift(1).fillna(initial_bankroll) * limited_kelly
        )
    else:
        raise ValueError(f"Unknown staking strategy: {strategy}")

    df["winner"] = pd.to_numeric(df["winner"], errors="coerce")
    df["profit"] = np.where(
        df["winner"] == 1, df["stake"] * (df["odds"] - 1), -df["stake"]
    )
    df["bankroll_end"] = (
        df["bankroll_start"].shift(1).fillna(initial_bankroll) + df["profit"]
    )

    # Drop the temporary start column for a cleaner output
    df = df.drop(columns=["bankroll_start"]).rename(
        columns={"bankroll_end": "bankroll"}
    )

    return df


def main_cli():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Simulate bankroll growth based on backtest results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--value-bets-csv",
        required=True,
        help="Path to the CSV file with value bet results.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save the detailed simulation results.",
    )
    parser.add_argument(
        "--strategy",
        choices=["flat", "percent", "kelly"],
        default="kelly",
        help="The staking strategy to use.",
    )
    parser.add_argument(
        "--stake-unit",
        type=float,
        default=1.0,
        help="For 'flat' strategy, the fixed stake amount. For 'percent', the percentage of bankroll to stake.",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.5,
        help="For 'kelly' strategy, the fraction of the Kelly stake to use (e.g., 0.5 for half-Kelly).",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=DEFAULT_INITIAL_BANKROLL,
        help="The starting bankroll for the simulation.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.value_bets_csv)
    simulation = simulate_bankroll_growth(
        df,
        initial_bankroll=args.initial_bankroll,
        strategy=args.strategy,
        stake_unit=args.stake_unit,
        kelly_fraction=args.kelly_fraction,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    simulation.to_csv(output_path, index=False)
    log_info(f"Saved full simulation results to {output_path}")

    if not simulation.empty:
        total_bets = len(simulation)
        final_bankroll = simulation["bankroll"].iloc[-1]
        total_profit = final_bankroll - args.initial_bankroll
        total_wagered = simulation["stake"].sum()
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        winning_bets = simulation[simulation["profit"] > 0]
        win_rate = (len(winning_bets) / total_bets) * 100 if total_bets > 0 else 0
        avg_odds = simulation["odds"].mean()
        peak_bankroll, max_drawdown = calculate_max_drawdown(simulation["bankroll"])

        summary = f"""
        --- Simulation Summary ---
        Strategy:           {args.strategy.title()}
        Initial Bankroll:   ${args.initial_bankroll:,.2f}
        Final Bankroll:     ${final_bankroll:,.2f}
        Peak Bankroll:      ${peak_bankroll:,.2f}
        
        Total Bets:         {total_bets:,}
        Winning Bets:       {len(winning_bets):,}
        Win Rate:           {win_rate:.2f}%
        
        Total Wagered:      ${total_wagered:,.2f}
        Net Profit:         ${total_profit:,.2f}
        Return on Capital:  {(total_profit / args.initial_bankroll) * 100:,.2f}%
        Return on Turnover: {roi:.2f}% (ROI)
        
        Average Odds:       {avg_odds:.2f}
        Max Drawdown:       {abs(max_drawdown):.2%}
        --------------------------
        """
        print(summary)


if __name__ == "__main__":
    main_cli()
