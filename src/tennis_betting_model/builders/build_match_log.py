# src/tennis_betting_model/builders/build_match_log.py
import pandas as pd
from pathlib import Path
import glob
import os
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    log_warning,
)
from tennis_betting_model.utils.common import get_surface
from tennis_betting_model.utils.config_schema import DataPaths
from tennis_betting_model.utils.schema import validate_data


def _create_historical_match_lookup(paths: DataPaths) -> pd.DataFrame:
    """
    Loads all historical ATP/WTA match files to create a lookup table for tournament names
    based on the date and players involved.
    Skips files with incompatible schemas.
    """
    log_info("Creating historical match lookup for tournament names...")
    raw_data_dir = Path(paths.raw_data_dir)
    all_matches = []

    use_cols = ["tourney_name", "tourney_date", "winner_id", "loser_id", "score"]

    for tour in ["atp", "wta"]:
        tour_files = glob.glob(
            os.path.join(raw_data_dir, f"tennis_{tour}", f"{tour}_matches_*.csv")
        )

        for f in tour_files:
            try:
                df_tour = pd.read_csv(f, usecols=use_cols, low_memory=False)
                all_matches.append(df_tour)
            except ValueError as e:
                if "Usecols do not match columns" in str(e):
                    log_warning(f"Skipping file with old schema: {Path(f).name}")
                else:
                    raise e

    if not all_matches:
        log_warning(
            "No historical match files with required columns found to build tournament name lookup."
        )
        return pd.DataFrame()

    df_historical = pd.concat(all_matches).dropna().drop_duplicates()

    df_historical["date"] = pd.to_datetime(
        df_historical["tourney_date"], format="%Y%m%d", errors="coerce"
    ).dt.date
    df_historical.dropna(subset=["date"], inplace=True)

    df_historical["p1_id"] = df_historical[["winner_id", "loser_id"]].min(axis=1)
    df_historical["p2_id"] = df_historical[["winner_id", "loser_id"]].max(axis=1)

    lookup = df_historical[["date", "p1_id", "p2_id", "tourney_name", "score"]].copy()
    lookup.rename(columns={"tourney_name": "historical_tourney_name"}, inplace=True)

    return lookup.drop_duplicates(subset=["date", "p1_id", "p2_id"], keep="last")


def main(paths: DataPaths):
    """
    Creates the historical match log and enriches it with tournament names and scores.
    """
    log_info("--- Creating Match Log from Summary File ---")

    log_info("Loading consolidated raw odds and player map...")
    raw_odds_path = Path(paths.betfair_raw_odds)
    map_path = Path(paths.player_map)

    if not raw_odds_path.exists() or not map_path.exists():
        log_error(
            "Raw odds or player map file not found. Run 'prepare-data' and 'create-player-map' first."
        )
        return

    # Load without dtype to avoid mypy issues
    df_raw = pd.read_csv(raw_odds_path, parse_dates=["tourney_date"])

    # Explicitly convert columns to string after loading
    df_raw["market_id"] = df_raw["market_id"].astype(str)
    df_raw["selection_id"] = df_raw["selection_id"].astype(str)

    df_raw["tourney_date"] = pd.to_datetime(df_raw["tourney_date"], utc=True)

    df_map = pd.read_csv(
        map_path, dtype={"betfair_id": "str", "historical_id": "Int64"}
    )

    log_info("Enriching summary data with historical IDs...")
    df_enriched = pd.merge(
        df_raw, df_map, left_on="selection_id", right_on="betfair_id", how="left"
    )

    df_settled = df_enriched[df_enriched["result"].isin(["WINNER", "LOSER"])].copy()
    df_settled.dropna(subset=["historical_id"], inplace=True)
    df_settled["historical_id"] = df_settled["historical_id"].astype(int)

    winners = df_settled[df_settled["result"] == "WINNER"].copy()
    losers = df_settled[df_settled["result"] == "LOSER"].copy()

    winners.rename(
        columns={
            "selection_id": "winner_id",
            "selection_name": "winner_name",
            "historical_id": "winner_historical_id",
        },
        inplace=True,
    )
    losers.rename(
        columns={
            "selection_id": "loser_id",
            "selection_name": "loser_name",
            "historical_id": "loser_historical_id",
        },
        inplace=True,
    )

    key_cols = ["market_id", "tourney_date", "competition_name"]
    match_log_df = pd.merge(
        winners[key_cols + ["winner_id", "winner_name", "winner_historical_id"]],
        losers[["market_id"] + ["loser_id", "loser_name", "loser_historical_id"]],
        on="market_id",
    )

    match_log_df.rename(
        columns={"competition_name": "tourney_name", "market_id": "match_id"},
        inplace=True,
    )

    df_match_lookup = _create_historical_match_lookup(paths)
    if not df_match_lookup.empty:
        match_log_df["date"] = match_log_df["tourney_date"].dt.date
        match_log_df["p1_id"] = match_log_df[
            ["winner_historical_id", "loser_historical_id"]
        ].min(axis=1)
        match_log_df["p2_id"] = match_log_df[
            ["winner_historical_id", "loser_historical_id"]
        ].max(axis=1)

        match_log_df = pd.merge(
            match_log_df, df_match_lookup, on=["date", "p1_id", "p2_id"], how="left"
        )

        nulls_before = match_log_df["tourney_name"].isnull().sum()
        match_log_df["tourney_name"] = match_log_df["tourney_name"].fillna(
            match_log_df["historical_tourney_name"]
        )
        nulls_after = match_log_df["tourney_name"].isnull().sum()
        log_success(
            f"Enriched tournament names. Nulls before: {nulls_before}, Nulls after: {nulls_after}"
        )

    match_log_df["surface"] = match_log_df["tourney_name"].apply(get_surface)

    match_log_df["sets_played"] = (
        match_log_df["score"].str.split().str.len().fillna(0).astype(int)
    )

    if match_log_df.empty:
        log_warning("⚠️ No valid, settled matches found. The match log will be empty.")
        output_path = Path(paths.betfair_match_log)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "match_id",
                "tourney_date",
                "tourney_name",
                "surface",
                "winner_id",
                "winner_historical_id",
                "winner_name",
                "loser_id",
                "loser_historical_id",
                "loser_name",
                "score",
                "sets_played",
            ]
        ).to_csv(output_path, index=False)
        return

    match_log_df.sort_values(by="tourney_date", inplace=True)

    final_cols = [
        "match_id",
        "tourney_date",
        "tourney_name",
        "surface",
        "winner_id",
        "winner_historical_id",
        "winner_name",
        "loser_id",
        "loser_historical_id",
        "loser_name",
        "score",
        "sets_played",
    ]
    for col in final_cols:
        if col not in match_log_df.columns:
            match_log_df[col] = None

    validated_df = validate_data(
        match_log_df[final_cols], "betfair_match_log", "Betfair Match Log"
    )

    output_path = Path(paths.betfair_match_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_path, index=False)
    log_success(
        f"Successfully created and validated match log with {len(validated_df)} entries."
    )
