import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    log_warning,
    setup_logging,
)


def create_match_log():
    """
    Processes a raw Betfair odds CSV to create a historical match log,
    now including the enriched historical_id.
    """
    setup_logging()
    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        input_path = Path(paths["betfair_odds"])
        output_path = Path(
            paths.get("betfair_match_log", "data/processed/betfair_match_log.csv")
        )
    except (FileNotFoundError, KeyError) as e:
        log_error(f"Could not load configuration. Error: {e}")
        return

    log_info(f"Loading raw Betfair odds data from: {input_path}")
    if not input_path.exists():
        log_error(f"Input file not found at {input_path}.")
        return

    try:
        df = pd.read_csv(input_path, parse_dates=["pt"])
    except Exception as e:
        log_error(f"Failed to load or parse the CSV file. Error: {e}")
        return

    log_info(
        f"Loaded {len(df)} records. Processing {df['market_id'].nunique()} unique markets."
    )

    log_info("Identifying the final state for each market...")
    latest_timestamps = df.groupby("market_id")["pt"].transform("max")
    df_final_state = df[df["pt"] == latest_timestamps].copy()

    runner_counts = df_final_state.groupby("market_id").size()
    valid_markets = runner_counts[runner_counts == 2].index
    df_two_runners = df_final_state[df_final_state["market_id"].isin(valid_markets)]

    log_info(
        f"Found {len(valid_markets)} markets with exactly two runners in their final state."
    )

    if df_two_runners.empty:
        log_warning("No valid two-runner markets found.")
        return

    match_results = []

    grouped = df_two_runners.groupby("market_id")
    for market_id, group in tqdm(grouped, desc="Determining Match Outcomes"):
        if len(group["runner_id"].unique()) != 2:
            continue

        winner = group.loc[group["best_back_price"].idxmin()]
        loser = group.loc[group.index.difference(winner.index)].iloc[0]

        if winner["runner_id"] == loser["runner_id"]:
            continue

        # --- FIX: Add historical_id to the output dictionary ---
        match_results.append(
            {
                "match_id": market_id,
                "tourney_date": winner["pt"].date(),
                "tourney_name": winner["tourney_name"],
                "winner_id": winner["runner_id"],
                "winner_historical_id": winner.get("historical_id"),
                "winner_name": winner["runner_name"],
                "loser_id": loser["runner_id"],
                "loser_historical_id": loser.get("historical_id"),
                "loser_name": loser["runner_name"],
            }
        )

    if not match_results:
        log_warning("Could not determine any match results from the data.")
        return

    match_log_df = pd.DataFrame(match_results)
    match_log_df.sort_values(by="tourney_date", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_log_df.to_csv(output_path, index=False)

    log_success(f"Successfully created match log with {len(match_log_df)} entries.")
    log_success(f"Match log saved to: {output_path}")


if __name__ == "__main__":
    create_match_log()
