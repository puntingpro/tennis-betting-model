# src/scripts/pipeline/match_selection_ids.py

import pandas as pd
from rapidfuzz import process

from scripts.utils.logger import log_info, log_warning
from scripts.utils.schema import enforce_schema, normalize_columns
from scripts.utils.selection import (
    build_market_runner_map,
    match_player_to_selection_id,
)

def find_fuzzy_match(name: str, choices: list[str], score_cutoff: int = 90) -> str | None:
    if not name or pd.isna(name):
        return None
    match = process.extractOne(name, choices, score_cutoff=score_cutoff)
    return match[0] if match else None

def assign_selection_ids(
    matches_df: pd.DataFrame, sackmann_df: pd.DataFrame, snapshots_df: pd.DataFrame
) -> pd.DataFrame:
    matches_df = normalize_columns(matches_df)
    sackmann_df = normalize_columns(sackmann_df)
    snapshots_df = normalize_columns(snapshots_df)

    market_map = build_market_runner_map(snapshots_df)
    unique_runner_names = snapshots_df["runner_name"].dropna().unique().tolist()

    log_info("Cleaning player names from historical data using fuzzy matching...")
    sackmann_df["player_1_clean"] = sackmann_df["winner_name"].apply(
        lambda x: find_fuzzy_match(x, unique_runner_names)
    )
    sackmann_df["player_2_clean"] = sackmann_df["loser_name"].apply(
        lambda x: find_fuzzy_match(x, unique_runner_names)
    )

    def find_market_id_for_cleaned_match(row):
        p1_clean = row["player_1_clean"]
        p2_clean = row["player_2_clean"]
        if not p1_clean or not p2_clean:
            return None
        p1_markets = set(snapshots_df[snapshots_df["runner_name"] == p1_clean]["market_id"])
        p2_markets = set(snapshots_df[snapshots_df["runner_name"] == p2_clean]["market_id"])
        common_markets = p1_markets.intersection(p2_markets)
        if common_markets:
            return common_markets.pop()
        log_warning(f"No common market found for cleaned players: {p1_clean} and {p2_clean}")
        return None

    log_info("Finding market IDs for cleaned player names...")
    sackmann_df["market_id"] = sackmann_df.apply(find_market_id_for_cleaned_match, axis=1)
    sackmann_df.dropna(subset=["market_id", "player_1_clean"], inplace=True)
    sackmann_df["match_id"] = sackmann_df["market_id"]

    # --- UPDATED SECTION ---
    # Ensure 'surface' is carried through the merge
    columns_to_merge = ["match_id", "player_1_clean", "player_2_clean", "tourney_date", "surface"]
    merged_df = matches_df.merge(
        sackmann_df[columns_to_merge], on="match_id", how="left"
    )
    # --- END UPDATED SECTION ---

    merged_df["winner"] = (merged_df["runner_name"] == merged_df["player_1_clean"]).astype(int)
    merged_df['player_1'] = merged_df['player_1_clean']
    merged_df['player_2'] = merged_df['player_2_clean']

    merged_df["selection_id_1"] = merged_df.apply(
        lambda r: match_player_to_selection_id(
            market_map, r["market_id"], r["player_1"]
        ),
        axis=1,
    )
    merged_df["selection_id_2"] = merged_df.apply(
        lambda r: match_player_to_selection_id(
            market_map, r["market_id"], r["player_2"]
        ),
        axis=1,
    )
    
    return enforce_schema(merged_df, "matches_with_ids")

def main_cli(args=None):
    log_info("This script is primarily used as a module within the main pipeline.")

if __name__ == "__main__":
    main_cli()