# src/scripts/pipeline/simulate_bankroll_growth.py

import sys
from pathlib import Path
import os

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
# --- End of fix ---

import numpy as np
import pandas as pd
import argparse

from src.scripts.utils.logger import log_info, log_success, setup_logging
from src.scripts.utils.schema import normalize_columns, patch_winner_column

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

    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        df = df.sort_values(by='tourney_date').reset_index(drop=True)

    df["bankroll"] = initial_bankroll
    df["strategy"] = strategy

    if strategy == "flat":
        df["stake"] = flat_stake_unit
    elif strategy == "kelly":
        df["kelly_fraction"] = pd.to_numeric(df["kelly_fraction"], errors='coerce').fillna(0)
        df["stake"] = df["bankroll"] * df["kelly_fraction"]
    else:
        raise ValueError(f"Unknown staking strategy: {strategy}")

    # Ensure the 'winner' column is numeric (0 or 1)
    df['winner'] = pd.to_numeric(df['winner'], errors='coerce')
    
    # Calculate profit/loss for each bet
    df["profit"] = np.where(
        df["winner"] == 1, df["stake"] * (df["odds"] - 1), -df["stake"]
    )
    df["bankroll"] = df["bankroll"].iloc[0] + df["profit"].cumsum()
    
    return df

def main_cli():
    setup_logging()
    parser = argparse.ArgumentParser(description="Simulate bankroll growth")
    parser.add_argument("--value_bets_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.value_bets_csv)
    simulation = simulate_bankroll_growth(df)
    
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    simulation.to_csv(output_path, index=False)
    log_info(f"Saved full simulation results to {output_path}")

    # --- NEW: Calculate and print summary ---
    if not simulation.empty:
        total_bets = len(simulation)
        final_bankroll = simulation['bankroll'].iloc[-1]
        total_profit = final_bankroll - 1000.0 # Assuming 1000 is the initial bankroll
        total_wagered = simulation['stake'].sum()
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        winning_bets = simulation[simulation['profit'] > 0]
        win_rate = (len(winning_bets) / total_bets) * 100 if total_bets > 0 else 0

        summary = f"""
        --- Simulation Summary ---
        Initial Bankroll:   $1,000.00
        Final Bankroll:     ${final_bankroll:,.2f}
        
        Total Bets:         {total_bets:,}
        Winning Bets:       {len(winning_bets):,}
        Win Rate:           {win_rate:.2f}%
        
        Total Wagered:      ${total_wagered:,.2f}
        Net Profit:         ${total_profit:,.2f}
        ROI:                {roi:.2f}%
        --------------------------
        """
        print(summary)
    # --- END NEW SECTION ---


if __name__ == "__main__":
    main_cli()