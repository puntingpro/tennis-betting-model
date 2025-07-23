# analyze_strategy.py

import pandas as pd


def analyze_profitability(csv_path="set_betting_analysis.csv", ev_threshold=0.1):
    """
    Loads the backtesting results and analyzes the profitability of a strategy
    based on a specific Expected Value (EV) threshold.
    """
    print("--- Strategy Analysis Report ---")
    print(f"Loading analysis data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        print("Please run the backtester.py first to generate this file.")
        return

    # --- 1. Isolate Potential Value Bets ---
    # Combine player 1 and player 2 bets into a single dataframe
    p1_bets = df[["market_id", "p1_id", "p1_odds", "p1_ev", "winner_id"]].rename(
        columns={"p1_id": "player_id", "p1_odds": "odds", "p1_ev": "ev"}
    )
    p2_bets = df[["market_id", "p2_id", "p2_odds", "p2_ev", "winner_id"]].rename(
        columns={"p2_id": "player_id", "p2_odds": "odds", "p2_ev": "ev"}
    )
    all_bets_df = pd.concat([p1_bets, p2_bets])

    # Filter for bets that meet our EV threshold
    value_bets_df = all_bets_df[all_bets_df["ev"] > ev_threshold].copy()

    if value_bets_df.empty:
        print(
            f"\nNo bets found with an EV greater than {ev_threshold:.2f}. Strategy did not place any bets."
        )
        return

    # --- 2. Calculate Real-World Performance ---
    # Determine the profit and loss for each bet (assuming a 1-unit stake)
    value_bets_df["won"] = value_bets_df["player_id"] == value_bets_df["winner_id"]
    value_bets_df["pnl"] = value_bets_df.apply(
        lambda row: (row["odds"] - 1) if row["won"] else -1, axis=1
    )

    # --- 3. Print the Final Report ---
    total_bets = len(value_bets_df)
    total_pnl = value_bets_df["pnl"].sum()
    roi = (total_pnl / total_bets) * 100
    win_rate = (value_bets_df["won"].sum() / total_bets) * 100
    avg_odds = value_bets_df["odds"].mean()

    print(f"\nStrategy: Bet on opportunities with EV > {ev_threshold:.2f}")
    print("-" * 30)
    print(f"Total Bets Placed: {total_bets}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Odds: {avg_odds:.2f}")
    print(f"Total Profit/Loss: {total_pnl:.2f} units")
    print(f"Return on Investment (ROI): {roi:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    # We can experiment with this threshold. 0.1 (10% EV) is a good starting point.
    analyze_profitability(ev_threshold=0.1)
