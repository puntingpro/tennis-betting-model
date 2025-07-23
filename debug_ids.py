import pandas as pd


def debug_id_mismatch():
    """
    Loads odds data and feature data to diagnose why they are not matching.
    """
    print("--- Starting ID Mismatch Debugger ---")

    # --- Load Odds Data ---
    try:
        odds_df = pd.read_csv("tennis_data.csv")
        print("Successfully loaded tennis_data.csv")
    except FileNotFoundError:
        print("Error: Could not find tennis_data.csv. Please ensure it exists.")
        return

    # --- Load Feature Data ---
    try:
        features_df = pd.read_csv("data/processed/all_advanced_features.csv")
        print("Successfully loaded data/processed/all_advanced_features.csv")
    except FileNotFoundError:
        print("Error: Could not find all_advanced_features.csv.")
        return

    # --- Extract Unique Player IDs from Both DataFrames ---
    odds_player_ids = set(odds_df["runner_id"].unique())
    features_p1_ids = set(features_df["p1_id"].unique())
    features_p2_ids = set(features_df["p2_id"].unique())
    all_feature_player_ids = features_p1_ids.union(features_p2_ids)

    print("\n--- Player ID Analysis ---")
    print(f"Found {len(odds_player_ids)} unique player IDs in the odds data.")
    print(f"Found {len(all_feature_player_ids)} unique player IDs in the feature data.")

    # --- Check for Overlap in Player IDs ---
    common_player_ids = odds_player_ids.intersection(all_feature_player_ids)
    print(f"Found {len(common_player_ids)} common player IDs between the two datasets.")

    if not common_player_ids:
        print(
            "CRITICAL: There are no common player IDs. The datasets cannot be matched."
        )
        # Optionally print some example IDs from each set to see the difference
        print("\nExample Player IDs from odds_df:", list(odds_player_ids)[:5])
        print("Example Player IDs from features_df:", list(all_feature_player_ids)[:5])
        return  # Stop here if no players match

    # --- Create a Unique, Order-Independent Match Identifier ---
    # We sort the player IDs within each match so (p1, p2) is the same as (p2, p1)
    def create_match_key(row):
        return tuple(sorted((row["p1_id"], row["p2_id"])))

    # Create a set of match keys from the features data
    feature_match_keys = set(features_df.apply(create_match_key, axis=1))

    # --- Check for Match Overlap ---
    odds_markets = odds_df.groupby("market_id")
    found_matches = 0

    print("\n--- Match Analysis ---")
    print(f"Analyzing {len(odds_markets)} markets from the odds data...")

    for market_id, market_df in odds_markets:
        player_ids = tuple(sorted(market_df["runner_id"].unique()))
        if len(player_ids) == 2:
            if player_ids in feature_match_keys:
                found_matches += 1

    print(
        f"Found {found_matches} markets in the odds data that have a corresponding entry in the feature data."
    )

    if found_matches == 0:
        print(
            "CRITICAL: Although some player IDs might match, no complete matchups could be found."
        )
        print("This confirms the reason the backtester processes no data.")
    else:
        print("SUCCESS: Found some matching markets. The issue may be more subtle.")


if __name__ == "__main__":
    debug_id_mismatch()
