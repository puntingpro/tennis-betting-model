import pandas as pd
import matplotlib.pyplot as plt


def analyze_results(
    csv_path="set_betting_analysis.csv",
):  # MODIFICATION: Default to the new set analysis file
    """
    Loads the detailed backtesting results and performs analysis.
    """
    print(f"Loading analysis data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        print("Please run the modified backtester.py first to generate this file.")
        return

    # --- 1. Analyze the Distribution of Expected Value (EV) ---
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    all_ev = pd.concat([df["p1_ev"], df["p2_ev"]])
    all_ev.hist(bins=100, ax=ax, alpha=0.8, label="EV for all bets")

    ax.set_title("Distribution of Expected Value (EV)", fontsize=16)
    ax.set_xlabel("Expected Value (EV)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.axvline(0, color="red", linestyle="--", label="Break-even (EV=0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ev_distribution.png")
    print("\nSaved EV distribution plot to ev_distribution.png")

    # --- 2. Compare Model Probability vs. Market's Implied Probability ---
    df["p1_implied_prob"] = 1 / df["p1_odds"]
    df["p2_implied_prob"] = 1 / df["p2_odds"]

    fig2, ax2 = plt.subplots(figsize=(10, 10))

    ax2.scatter(df["p1_implied_prob"], df["p1_model_prob"], alpha=0.3, label="Player 1")
    ax2.scatter(df["p2_implied_prob"], df["p2_model_prob"], alpha=0.3, label="Player 2")

    ax2.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Agreement")

    ax2.set_title("Model Probability vs. Market Implied Probability", fontsize=16)
    ax2.set_xlabel("Market Implied Probability (1 / Odds)", fontsize=12)
    ax2.set_ylabel("Model's Predicted Probability", fontsize=12)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("probability_comparison.png")
    print("Saved probability comparison plot to probability_comparison.png")

    # --- 3. Print Summary Statistics ---
    print("\n--- Summary Statistics ---")
    print("\nExpected Value (EV):")
    print(all_ev.describe())

    p1_prob_diff = (df["p1_model_prob"] - df["p1_implied_prob"]).abs()
    p2_prob_diff = (df["p2_model_prob"] - df["p2_implied_prob"]).abs()

    print("\nAverage Absolute Difference between Model and Market Probabilities:")
    print(f"Player 1: {p1_prob_diff.mean():.4f}")
    print(f"Player 2: {p2_prob_diff.mean():.4f}")

    plt.show()


if __name__ == "__main__":
    analyze_results()
