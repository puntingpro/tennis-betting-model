import pandas as pd


def generate_id_map():
    """
    Creates a mapping file between Betfair runner_ids and the feature data p_ids
    by matching on player names.
    """
    print("--- Starting Player ID Map Creation ---")

    # --- 1. Load Betfair Odds Data and Extract Player Names ---
    try:
        odds_df = pd.read_csv("tennis_data.csv", usecols=["runner_id", "runner_name"])
        odds_df.drop_duplicates(inplace=True)
        odds_df.dropna(subset=["runner_name"], inplace=True)
        print(
            f"Loaded {len(odds_df)} unique player/ID combinations from tennis_data.csv"
        )
    except FileNotFoundError:
        print("Error: Could not find tennis_data.csv.")
        return

    # --- 2. Load Features Data and Extract Player Names ---
    try:
        # We need to load all player-related columns
        features_df = pd.read_csv(
            "data/processed/all_advanced_features.csv",
            usecols=["p1_id", "p1_name", "p2_id", "p2_name"],
        )
        print("Loaded features from data/processed/all_advanced_features.csv")
    except FileNotFoundError:
        print("Error: Could not find all_advanced_features.csv.")
        return

    # Normalize the feature data into a single list of (id, name)
    p1_data = features_df[["p1_id", "p1_name"]].rename(
        columns={"p1_id": "feature_id", "p1_name": "player_name"}
    )
    p2_data = features_df[["p2_id", "p2_name"]].rename(
        columns={"p2_id": "feature_id", "p2_name": "player_name"}
    )

    all_feature_players = pd.concat([p1_data, p2_data])
    all_feature_players.drop_duplicates(subset=["feature_id"], inplace=True)
    all_feature_players.dropna(inplace=True)
    print(
        f"Extracted {len(all_feature_players)} unique players from the features file."
    )

    # --- 3. Merge the two DataFrames on Player Name ---
    # We rename the columns to be clear about their origin
    odds_df.rename(
        columns={"runner_id": "betfair_id", "runner_name": "player_name"}, inplace=True
    )

    # Perform the merge (the 'inner' join means we only keep names that appear in BOTH files)
    player_map = pd.merge(odds_df, all_feature_players, on="player_name", how="inner")

    if player_map.empty:
        print(
            "\nCRITICAL: Could not find any common player names between the two files."
        )
        print(
            "Please check for inconsistencies in names (e.g., 'A. Murray' vs 'Andy Murray')."
        )
        return

    # --- 4. Save the Mapping File ---
    output_path = "player_id_map.csv"
    final_map = player_map[["betfair_id", "feature_id", "player_name"]]
    final_map.to_csv(output_path, index=False)

    print(f"\nSUCCESS: Created player ID map with {len(final_map)} entries.")
    print(f"Map saved to {output_path}")
    print("\nNext step is to modify backtester.py to use this map.")


if __name__ == "__main__":
    generate_id_map()
