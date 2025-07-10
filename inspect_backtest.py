import pandas as pd
import argparse
import glob
from pathlib import Path
import yaml
import sys # <-- The missing import

# --- Add project root to the Python path to allow importing from src ---
# This assumes the script is run from the root of the project
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

def inspect_backtest(target_tournament: str, config_path: str):
    """
    Loads backtest and feature data to inspect the details of bets for a specific tournament.
    """
    # --- Load config to get file paths ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        backtest_file = config['data_paths']['backtest_results']
        features_file = config['data_paths']['consolidated_features']
    except (FileNotFoundError, KeyError):
        print(f"Error: Could not load required paths from '{config_path}'.")
        return

    # --- Load both the backtest results and the original features ---
    try:
        df_bets = pd.read_csv(backtest_file)
        df_features = pd.read_csv(features_file, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure you have run both the 'build' and 'backtest' commands.")
        return

    # --- Merge the bets with their corresponding features ---
    # Pandas adds suffixes (_x, _y) to columns with the same name during a merge.
    df = pd.merge(df_bets, df_features, on='match_id', how='left')

    print(f"--- Analysis for Tournament: {target_tournament} ---")

    # Use the suffixed column name 'tourney_name_x' for filtering
    tournament_df = df[df["tourney_name_x"] == target_tournament].copy()

    if tournament_df.empty:
        print("No bets found for this tournament.")
    else:
        # --- Calculate and print summary statistics ---
        num_bets = len(tournament_df)
        
        # 'winner_x' is from the bets file, 'winner_y' is the true outcome from the features file
        tournament_df['is_correct'] = tournament_df['winner_x'] == tournament_df['winner_y']
        
        tournament_df['profit'] = tournament_df.apply(
            lambda row: (row['odds'] - 1) if row['is_correct'] else -1,
            axis=1
        )
        
        total_profit = tournament_df['profit'].sum()
        roi = (total_profit / num_bets) * 100 if num_bets > 0 else 0
        avg_odds = tournament_df['odds'].mean()
        win_rate = (tournament_df['is_correct'].sum() / num_bets) * 100 if num_bets > 0 else 0

        print(f"\nSummary Stats:")
        print(f"  - Total Bets: {num_bets}")
        print(f"  - Average Odds: {avg_odds:.2f}")
        print(f"  - Win Rate: {win_rate:.2f}%")
        print(f"  - Total Profit: {total_profit:.2f} units")
        print(f"  - ROI: {roi:.2f}%")

        # --- Display the individual bets along with key features ---
        features_to_show = [
            'match_id', 'odds', 'predicted_prob', 'profit',
            'p1_rank', 'p2_rank', 'p1_win_perc', 'p2_win_perc',
            'h2h_p1_wins', 'h2h_p2_wins'
        ]
        
        # Ensure we only try to display columns that actually exist
        features_to_show = [f for f in features_to_show if f in tournament_df.columns]

        print("\nIndividual Bets and Key Features:")
        print(tournament_df[features_to_show].to_string(index=False))

    print("\n--- Analysis Complete ---")

def main():
    """
    Main function to parse arguments and run the inspection.
    """
    parser = argparse.ArgumentParser(
        description="Inspect the features of bets for a specific tournament.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("tournament_name", type=str, help="The exact name of the tournament to inspect (use quotes if it contains spaces).")
    parser.add_argument("--config", default="config.yaml", help="Path to the project's config file.")
    args = parser.parse_args()

    inspect_backtest(args.tournament_name, args.config)

if __name__ == "__main__":
    main()