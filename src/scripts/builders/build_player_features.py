# src/scripts/builders/build_player_features.py

from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

from src.scripts.utils.config import load_config
from src.scripts.utils.logger import setup_logging, log_info, log_success
from src.scripts.utils.schema import validate_data, PlayerFeaturesSchema
from src.scripts.utils.constants import DEFAULT_PLAYER_RANK, ELO_INITIAL_RATING


def calculate_player_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Calculates point-in-time stats for each player in each match."""
    id_vars = ["match_id", "tourney_date", "surface"]
    p1_cols = ["winner_id", "loser_id"]
    p2_cols = ["loser_id", "winner_id"]

    df_p1 = df_matches[id_vars + p1_cols].rename(
        columns={"winner_id": "player_id", "loser_id": "opponent_id"}
    )
    df_p1["won"] = 1

    df_p2 = df_matches[id_vars + p2_cols].rename(
        columns={"loser_id": "player_id", "winner_id": "opponent_id"}
    )
    df_p2["won"] = 0

    df_player_matches = pd.concat([df_p1, df_p2], ignore_index=True)
    df_player_matches = df_player_matches.sort_values(by="tourney_date")

    gb_player = df_player_matches.groupby("player_id")
    df_player_matches["matches_played"] = gb_player.cumcount()
    df_player_matches["wins"] = gb_player["won"].cumsum() - df_player_matches["won"]

    gb_surface = df_player_matches.groupby(["player_id", "surface"])
    df_player_matches["surface_matches"] = gb_surface.cumcount()
    df_player_matches["surface_wins"] = (
        gb_surface["won"].cumsum() - df_player_matches["won"]
    )

    df_player_matches["form_last_10"] = (
        gb_player["won"].shift(1).rolling(window=10, min_periods=1).mean().fillna(0)
    )
    df_player_matches["win_perc"] = (
        df_player_matches["wins"] / df_player_matches["matches_played"]
    ).fillna(0)
    df_player_matches["surface_win_perc"] = (
        df_player_matches["surface_wins"] / df_player_matches["surface_matches"]
    ).fillna(0)

    stats_cols = [
        "match_id",
        "player_id",
        "win_perc",
        "surface_win_perc",
        "form_last_10",
    ]
    return df_player_matches[stats_cols]


def add_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates point-in-time Head-to-Head (H2H) stats."""
    log_info("Calculating Head-to-Head (H2H) stats...")
    h2h_records: defaultdict[tuple[int, int], defaultdict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    h2h_results = []
    df_sorted = df.sort_values(by="tourney_date")
    for row in tqdm(df_sorted.itertuples(), total=len(df), desc="H2H Calculation"):
        # FIX: Ignore the strict mypy error for itertuples.
        p1_id, p2_id = int(row.p1_id), int(row.p2_id)  # type: ignore

        # Be explicit about sorting the tuple to satisfy MyPy.
        player_pair: tuple[int, int] = (
            (p1_id, p2_id) if p1_id < p2_id else (p2_id, p1_id)
        )

        p1_wins_vs_p2 = h2h_records[player_pair][p1_id]
        p2_wins_vs_p1 = h2h_records[player_pair][p2_id]
        h2h_results.append(
            {
                "match_id": row.match_id,
                "h2h_p1_wins": p1_wins_vs_p2,
                "h2h_p2_wins": p2_wins_vs_p1,
            }
        )
        # FIX: Ignore the strict mypy error for itertuples.
        winner_id = int(row.winner_id)  # type: ignore
        h2h_records[player_pair][winner_id] += 1
    h2h_df = pd.DataFrame(h2h_results)
    return pd.merge(df, h2h_df, on="match_id", how="left")


def main(args):
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]

    log_info("Loading consolidated data and other required files...")
    df_matches = pd.read_csv(paths["consolidated_matches"])
    df_rankings = pd.read_csv(paths["consolidated_rankings"])
    df_players = pd.read_csv(paths["raw_players"], encoding="latin-1")
    df_elo = pd.read_csv(paths["elo_ratings"])

    log_info("Preprocessing data...")
    df_matches["tourney_date"] = pd.to_datetime(
        df_matches["tourney_date"], errors="coerce"
    ).dt.tz_localize("UTC")
    df_rankings["ranking_date"] = pd.to_datetime(
        df_rankings["ranking_date"], utc=True, errors="coerce"
    )
    df_matches.dropna(subset=["tourney_date", "winner_id", "loser_id"], inplace=True)
    df_rankings.dropna(subset=["ranking_date", "player"], inplace=True)

    # Ensure these columns are integer types before processing.
    df_matches["winner_id"] = df_matches["winner_id"].astype("int64")
    df_matches["loser_id"] = df_matches["loser_id"].astype("int64")
    df_rankings["player"] = df_rankings["player"].astype("int64")

    if "match_id" not in df_matches.columns:
        df_matches["match_id"] = (
            df_matches["tourney_id"].astype(str)
            + "-"
            + df_matches["match_num"].astype(str)
        )
    df_matches = df_matches.sort_values(by="tourney_date")

    # --- BUG FIX: Randomly assign p1 and p2 to prevent data leakage ---
    log_info("Randomly assigning P1/P2 to prevent data leakage...")
    is_swapped = np.random.choice([True, False], size=len(df_matches))

    p1_ids = np.where(is_swapped, df_matches["loser_id"], df_matches["winner_id"])
    p2_ids = np.where(is_swapped, df_matches["winner_id"], df_matches["loser_id"])

    features_df = df_matches[
        ["match_id", "tourney_date", "tourney_name", "surface", "winner_id"]
    ].copy()
    features_df["p1_id"] = p1_ids
    features_df["p2_id"] = p2_ids
    # --- END FIX ---

    log_info("Calculating player stats (vectorized)...")
    player_stats_df = calculate_player_stats(df_matches)

    log_info("Merging stats and building final feature set...")
    features_df = add_h2h_stats(features_df)
    features_df["winner"] = (features_df["p1_id"] == features_df["winner_id"]).astype(
        int
    )
    features_df = features_df.drop(columns=["winner_id"])

    log_info("Merging Elo ratings...")
    features_df = pd.merge(
        features_df,
        df_elo,
        left_on=["match_id"],
        right_on=["match_id"],
        how="left",
    )

    features_df["p1_elo"] = np.where(
        features_df["p1_id"] == features_df["p1_id_elo"],
        features_df["p1_elo"],
        features_df["p2_elo"],
    )
    features_df["p2_elo"] = np.where(
        features_df["p2_id"] == features_df["p2_id_elo"],
        features_df["p2_elo"],
        features_df["p1_elo"],
    )
    features_df.drop(columns=["p1_id_elo", "p2_id_elo"], inplace=True)

    features_df["p1_elo"].fillna(ELO_INITIAL_RATING, inplace=True)
    features_df["p2_elo"].fillna(ELO_INITIAL_RATING, inplace=True)

    p1_stats = player_stats_df.rename(
        columns={
            "win_perc": "p1_win_perc",
            "surface_win_perc": "p1_surface_win_perc",
            "form_last_10": "p1_form_last_10",
        }
    )
    features_df = pd.merge(
        features_df,
        p1_stats,
        left_on=["match_id", "p1_id"],
        right_on=["match_id", "player_id"],
        how="left",
    ).drop("player_id", axis=1)

    p2_stats = player_stats_df.rename(
        columns={
            "win_perc": "p2_win_perc",
            "surface_win_perc": "p2_surface_win_perc",
            "form_last_10": "p2_form_last_10",
        }
    )
    features_df = pd.merge(
        features_df,
        p2_stats,
        left_on=["match_id", "p2_id"],
        right_on=["match_id", "player_id"],
        how="left",
    ).drop("player_id", axis=1)

    player_info = df_players[["player_id", "hand", "height"]].set_index("player_id")
    features_df = features_df.merge(
        player_info, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"hand": "p1_hand", "height": "p1_height"})
    features_df = features_df.merge(
        player_info, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"hand": "p2_hand", "height": "p2_height"})

    log_info("Fetching player rankings (vectorized)...")
    df_rankings_sorted = df_rankings.sort_values(by="ranking_date")
    features_df_sorted = features_df.sort_values(by="tourney_date")

    p1_ranks = pd.merge_asof(
        left=features_df_sorted[["match_id", "tourney_date", "p1_id"]],
        right=df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="p1_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "p1_rank"})[["match_id", "p1_rank"]]
    p2_ranks = pd.merge_asof(
        left=features_df_sorted[["match_id", "tourney_date", "p2_id"]],
        right=df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="p2_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "p2_rank"})[["match_id", "p2_rank"]]

    features_df = pd.merge(features_df, p1_ranks, on="match_id", how="left")
    features_df = pd.merge(features_df, p2_ranks, on="match_id", how="left")

    features_df["p1_rank"].fillna(DEFAULT_PLAYER_RANK, inplace=True)
    features_df["p2_rank"].fillna(DEFAULT_PLAYER_RANK, inplace=True)

    features_df["rank_diff"] = features_df["p1_rank"] - features_df["p2_rank"]
    features_df["elo_diff"] = features_df["p1_elo"] - features_df["p2_elo"]

    features_df = validate_data(features_df, PlayerFeaturesSchema(), "player_features")

    output_path = Path(paths["consolidated_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving features to {output_path}...")
    features_df.to_csv(output_path, index=False)

    log_success(f"✅ Successfully created feature library at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)
