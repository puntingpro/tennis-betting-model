# debug_build_inputs.py

import pandas as pd
from pathlib import Path
from tennis_betting_model.utils.config import load_config

print("--- Running Build Input Diagnostics ---")

try:
    config = load_config("config.yaml")
    paths = config["data_paths"]

    # --- Check 1: Consolidated Matches ---
    print("\n--- Checking: consolidated_matches.csv ---")
    matches_path = Path(paths["consolidated_matches"])
    if matches_path.exists():
        df_matches = pd.read_csv(matches_path)
        print(f"Found {len(df_matches)} rows.")
        print("Sample:")
        print(df_matches.head().to_string())
    else:
        print("File not found.")

    # --- Check 2: Consolidated Rankings ---
    print("\n--- Checking: consolidated_rankings.csv ---")
    rankings_path = Path(paths["consolidated_rankings"])
    if rankings_path.exists():
        df_rankings = pd.read_csv(rankings_path)
        print(f"Found {len(df_rankings)} rows.")
        print("Sample:")
        print(df_rankings.head().to_string())
    else:
        print("File not found.")

    # --- Check 3: Raw Players ---
    print("\n--- Checking: raw_players.csv ---")
    players_path = Path(paths["raw_players"])
    if players_path.exists():
        df_players = pd.read_csv(players_path, encoding="latin-1")
        print(f"Found {len(df_players)} rows.")
        print("Sample:")
        print(df_players.head().to_string())
    else:
        print("File not found.")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Diagnostics Complete ---")
