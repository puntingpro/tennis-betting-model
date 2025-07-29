# create_mapping_file.py

import pandas as pd
import glob
import os
from pathlib import Path
from thefuzz import process
from tqdm import tqdm

print("--- Player Mapping File Generator ---")

# Load unique players from Betfair Data
print("Loading unique players from Betfair data...")
betfair_odds_path = Path("./data/processed/betfair_set_winner_odds.csv.bak")
df_betfair = pd.read_csv(betfair_odds_path)
betfair_players = df_betfair[["runner_id", "runner_name"]].drop_duplicates().dropna()
betfair_players = betfair_players[betfair_players["runner_name"].str.strip() != ""]
print(f"Found {len(betfair_players)} unique players in Betfair data.")

# Load unique players from Historical Data
print("Loading unique players from historical data...")
raw_data_dir = Path("./data/raw")
tour_types = [
    "atp_matches_[1-2][0-9][0-9][0-9].csv",
    "atp_matches_qual_chall_*.csv",
    "atp_matches_futures_*.csv",
]
atp_files = [
    f
    for p in tour_types
    for f in glob.glob(os.path.join(raw_data_dir, "tennis_atp", p))
]
df_historical_list = [pd.read_csv(f, low_memory=False) for f in atp_files]
df_historical = pd.concat(df_historical_list, ignore_index=True)

historical_winners = df_historical[["winner_id", "winner_name"]].rename(
    columns={"winner_id": "historical_id", "winner_name": "historical_name"}
)
historical_losers = df_historical[["loser_id", "loser_name"]].rename(
    columns={"loser_id": "historical_id", "loser_name": "historical_name"}
)
historical_players = (
    pd.concat([historical_winners, historical_losers]).drop_duplicates().dropna()
)
historical_players = historical_players[
    historical_players["historical_name"].str.strip() != ""
]
print(f"Found {len(historical_players)} unique players in historical data.")

# --- Perform Fuzzy Matching ---
print("Performing fuzzy string matching to find likely pairs...")
historical_name_map = historical_players.set_index("historical_name")["historical_id"]
historical_name_list = historical_name_map.index.tolist()

mappings = []
for index, row in tqdm(betfair_players.iterrows(), total=len(betfair_players)):
    betfair_id = row["runner_id"]
    betfair_name = row["runner_name"]

    # Find the best match from the historical list
    best_match, score = process.extractOne(betfair_name, historical_name_list)

    if score >= 85:  # Use a confidence threshold of 85
        historical_id = historical_name_map[best_match]
        mappings.append(
            {
                "betfair_id": betfair_id,
                "historical_id": historical_id,
                "betfair_name": betfair_name,
                "matched_name": best_match,
                "confidence": score,
            }
        )

mapping_df = pd.DataFrame(mappings)
output_path = Path("./player_mapping.csv")
mapping_df.to_csv(output_path, index=False)

print("\n--- Mapping Complete ---")
print(
    f"Successfully generated a candidate mapping file with {len(mapping_df)} entries."
)
print(f"File saved to: {output_path}")
print("\nIMPORTANT: Please manually review and correct this file before use.")
