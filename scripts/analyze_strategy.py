# analyze_strategy.py

import pandas as pd


def analyze_profitability(csv_path="set_betting_analysis.csv"):
    """
    Loads the backtesting results and performs a deep analysis of profitability,
    focusing on the final, combined strategy.
    """
    print("--- Definitive Strategy Analysis Report ---")
    print(f"Loading analysis data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        print("Please run the backtester.py first to generate this file.")
        return

    # Combine player 1 and player 2 bets into a single dataframe
    p1_bets = df[["market_id", "p1_id", "p1_odds", "p1_ev", "winner_id"]].rename(
        columns={"p1_id": "player_id", "p1_odds": "odds", "p1_ev": "ev"}
    )
    p2_bets = df[["market_id", "p2_id", "p2_odds", "p2_ev", "winner_id"]].rename(
        columns={"p2_id": "player_id", "p2_odds": "odds", "p2_ev": "ev"}
    )
    all_bets_df = pd.concat([p1_bets, p2_bets])

    # Filter for only bets with a positive EV to start
    value_bets_df = all_bets_df[all_bets_df["ev"] > 0].copy()

    if value_bets_df.empty:
        print("\nNo bets found with a positive EV. No analysis to perform.")
        return

    # Determine PnL for each bet
    value_bets_df["won"] = value_bets_df["player_id"] == value_bets_df["winner_id"]
    value_bets_df["pnl"] = value_bets_df.apply(
        lambda row: (row["odds"] - 1) if row["won"] else -1, axis=1
    )

    # --- FINAL STRATEGY ANALYSIS ---
    print("\n--- Analysis of Potential Winning Strategies ---")

    # 1. Underdog Strategy
    underdog_strategy_df = value_bets_df[value_bets_df["odds"] >= 2.5]
    print_report(underdog_strategy_df, "Strategy 1: Bet on Underdogs (Odds >= 2.5)")

    # 2. High EV Strategy
    high_ev_strategy_df = value_bets_df[value_bets_df["ev"] > 0.3]
    print_report(high_ev_strategy_df, "Strategy 2: Bet on High EV Signals (EV > 0.30)")

    # 3. Combined "Sweet Spot" Strategy
    combined_strategy_df = value_bets_df[
        (value_bets_df["odds"] >= 2.5) & (value_bets_df["ev"] > 0.30)
    ]
    print_report(
        combined_strategy_df, "Strategy 3 (Combined): Bet on High EV Underdogs"
    )


def print_report(df, title):
    """Helper function to print a standardized performance report."""
    if df.empty:
        print(f"\n{title}\n" + "-" * 40 + "\nNo bets in this category.\n" + "-" * 40)
        return

    total_bets = len(df)
    total_pnl = df["pnl"].sum()
    roi = (total_pnl / total_bets) * 100
    win_rate = (df["won"].sum() / total_bets) * 100
    avg_odds = df["odds"].mean()

    print(f"\n{title}")
    print("-" * 40)
    print(f"Total Bets Placed: {total_bets}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Total Profit/Loss: {total_pnl:.2f} units")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    print("-" * 40)


if __name__ == "__main__":
    analyze_profitability()
