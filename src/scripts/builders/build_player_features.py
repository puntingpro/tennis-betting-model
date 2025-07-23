# src/scripts/builders/build_player_features.py

from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

from scripts.utils.config import load_config
from scripts.utils.logger import setup_logging, log_info, log_success
from scripts.utils.constants import DEFAULT_PLAYER_RANK, ELO_INITIAL_RATING


def parse_score_to_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the match score string to determine the winner of each set.
    Expands the DataFrame to have one row per set.
    """
    log_info("Parsing match scores to determine set winners...")
    sets_data = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Parsing Scores"):
        try:
            scores = str(row.score).split(" ")
            for i, set_score in enumerate(scores):
                if "-" in set_score and "RET" not in set_score:
                    p1_games, p2_games = map(int, set_score.split("-"))

                    set_num = i + 1
                    set_winner_id = (
                        row.winner_id if p1_games > p2_games else row.loser_id
                    )

                    sets_data.append(
                        {
                            "match_id": row.match_id,
                            "set_num": set_num,
                            "set_winner_id": set_winner_id,
                        }
                    )
        except (ValueError, IndexError):
            continue

    sets_df = pd.DataFrame(sets_data)
    return pd.merge(df, sets_df, on="match_id", how="inner")


def calculate_surface_elo(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates point-in-time, surface-specific Elo ratings for each player.
    """
    log_info("Calculating surface-specific Elo ratings...")
    surface_elos = defaultdict(lambda: defaultdict(lambda: ELO_INITIAL_RATING))

    elo_results = []

    for row in tqdm(
        df_matches.itertuples(), total=len(df_matches), desc="Elo Calculation"
    ):
        winner_id = row.winner_id
        loser_id = row.loser_id
        surface = row.surface

        winner_elo = surface_elos[winner_id][surface]
        loser_elo = surface_elos[loser_id][surface]

        expected_win_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        k = 32
        new_winner_elo = winner_elo + k * (1 - expected_win_winner)
        new_loser_elo = loser_elo + k * (0 - (1 - expected_win_winner))

        surface_elos[winner_id][surface] = new_winner_elo
        surface_elos[loser_id][surface] = new_loser_elo

        # Ensure consistent p1/p2 assignment for merging
        p1_id = min(winner_id, loser_id)
        p2_id = max(winner_id, loser_id)

        elo_results.append(
            {
                "match_id": row.match_id,
                "p1_id": p1_id,
                "p2_id": p2_id,
                "p1_surface_elo_pre_match": surface_elos[p1_id][surface],
                "p2_surface_elo_pre_match": surface_elos[p2_id][surface],
            }
        )

    return pd.DataFrame(elo_results)


def calculate_player_stats(df_matches_with_ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates point-in-time stats with anti-leakage logic.
    """
    log_info("Calculating player stats with anti-leakage logic...")

    p1_cols = {
        "winner_id": "player_id",
        "loser_id": "opponent_id",
        "winner_rank": "player_rank",
        "loser_rank": "opponent_rank",
    }
    p2_cols = {
        "loser_id": "player_id",
        "winner_id": "opponent_id",
        "loser_rank": "player_rank",
        "winner_rank": "opponent_rank",
    }

    p1 = df_matches_with_ranks[
        [
            "match_id",
            "tourney_date",
            "surface",
            "winner_id",
            "loser_id",
            "score",
            "winner_rank",
            "loser_rank",
        ]
    ].rename(columns=p1_cols)
    p1["won"] = 1
    p2 = df_matches_with_ranks[
        [
            "match_id",
            "tourney_date",
            "surface",
            "winner_id",
            "loser_id",
            "score",
            "winner_rank",
            "loser_rank",
        ]
    ].rename(columns=p2_cols)
    p2["won"] = 0

    player_matches = pd.concat([p1, p2]).sort_values(by="tourney_date")

    player_stats_cache = defaultdict(
        lambda: {
            "matches_played": 0,
            "wins": 0,
            "form": [],
            "surface_matches": defaultdict(int),
            "surface_wins": defaultdict(int),
            "last_match_date": pd.NaT,
            "recent_sets": [],
            "opponent_ranks": [],
        }
    )

    match_features = []

    for date, daily_matches in tqdm(
        player_matches.groupby("tourney_date"), desc="Processing matches day-by-day"
    ):
        daily_features = []
        for row in daily_matches.itertuples():
            player_id = row.player_id
            stats = player_stats_cache[player_id]

            daily_features.append(
                {
                    "match_id": row.match_id,
                    "player_id": player_id,
                    "win_perc": (stats["wins"] / stats["matches_played"])
                    if stats["matches_played"] > 0
                    else 0,
                    "surface_win_perc": (
                        stats["surface_wins"][row.surface]
                        / stats["surface_matches"][row.surface]
                    )
                    if stats["surface_matches"][row.surface] > 0
                    else 0,
                    "form_last_10": np.mean(stats["form"]) if stats["form"] else 0,
                    "rest_days": (date - stats["last_match_date"]).days
                    if pd.notna(stats["last_match_date"])
                    else 30,
                    "sets_played_last_7d": sum(
                        count
                        for match_date, count in stats["recent_sets"]
                        if (date - match_date).days <= 7
                    ),
                    "avg_opponent_rank_last_10": np.mean(stats["opponent_ranks"])
                    if stats["opponent_ranks"]
                    else DEFAULT_PLAYER_RANK,
                }
            )

        match_features.extend(daily_features)

        for row in daily_matches.itertuples():
            stats = player_stats_cache[row.player_id]
            stats["matches_played"] += 1
            stats["wins"] += row.won
            stats["surface_matches"][row.surface] += 1
            stats["surface_wins"][row.surface] += row.won
            stats["form"] = (stats["form"] + [row.won])[-10:]
            stats["opponent_ranks"] = (stats["opponent_ranks"] + [row.opponent_rank])[
                -10:
            ]
            stats["last_match_date"] = date
            try:
                sets_played = len(str(row.score).split())
                stats["recent_sets"] = [
                    s for s in stats["recent_sets"] if (date - s[0]).days <= 30
                ] + [(date, sets_played)]
            except:
                continue

    return pd.DataFrame(match_features)


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
    df_matches.dropna(
        subset=["tourney_date", "winner_id", "loser_id", "score"], inplace=True
    )
    df_rankings.dropna(subset=["ranking_date", "player"], inplace=True)
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

    df_rankings_sorted = df_rankings.sort_values(by="ranking_date")
    winner_ranks = pd.merge_asof(
        left=df_matches[["tourney_date", "winner_id"]],
        right=df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="winner_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "winner_rank"})
    loser_ranks = pd.merge_asof(
        left=df_matches[["tourney_date", "loser_id"]],
        right=df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="loser_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "loser_rank"})
    df_matches["winner_rank"] = winner_ranks["winner_rank"].fillna(DEFAULT_PLAYER_RANK)
    df_matches["loser_rank"] = loser_ranks["loser_rank"].fillna(DEFAULT_PLAYER_RANK)

    player_stats_df = calculate_player_stats(df_matches)
    df_surface_elo = calculate_surface_elo(df_matches)
    df_sets = parse_score_to_sets(df_matches)

    log_info("Assembling final feature set...")
    features_df = df_sets[["match_id", "set_num", "set_winner_id"]]

    # Assign P1/P2 consistently
    match_players = df_matches[["match_id", "winner_id", "loser_id"]].copy()
    match_players["p1_id"] = np.minimum(
        match_players["winner_id"], match_players["loser_id"]
    )
    match_players["p2_id"] = np.maximum(
        match_players["winner_id"], match_players["loser_id"]
    )
    features_df = pd.merge(
        features_df, match_players[["match_id", "p1_id", "p2_id"]], on="match_id"
    )

    features_df["winner"] = (
        features_df["p1_id"] == features_df["set_winner_id"]
    ).astype(int)

    # Merge all features
    features_df = pd.merge(features_df, df_elo, on="match_id", how="left")
    features_df = pd.merge(
        features_df, df_surface_elo, on=["match_id", "p1_id", "p2_id"], how="left"
    )

    p1_stats = player_stats_df.rename(
        columns=lambda c: f"p1_{c}" if c not in ["match_id", "player_id"] else c
    )
    features_df = pd.merge(
        features_df,
        p1_stats,
        left_on=["match_id", "p1_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    p2_stats = player_stats_df.rename(
        columns=lambda c: f"p2_{c}" if c not in ["match_id", "player_id"] else c
    )
    features_df = pd.merge(
        features_df,
        p2_stats,
        left_on=["match_id", "p2_id"],
        right_on=["match_id", "player_id"],
        how="left",
    )

    player_info = df_players[["player_id", "hand", "height"]].set_index("player_id")
    features_df = features_df.merge(
        player_info, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"hand": "p1_hand", "height": "p1_height"})
    features_df = features_df.merge(
        player_info, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"hand": "p2_hand", "height": "p2_height"})

    p1_ranks = df_matches[
        ["match_id", "winner_id", "loser_id", "winner_rank", "loser_rank"]
    ].copy()
    p1_ranks["p1_id"] = np.minimum(p1_ranks["winner_id"], p1_ranks["loser_id"])
    p1_ranks["p1_rank"] = np.where(
        p1_ranks["p1_id"] == p1_ranks["winner_id"],
        p1_ranks["winner_rank"],
        p1_ranks["loser_rank"],
    )

    p2_ranks = df_matches[
        ["match_id", "winner_id", "loser_id", "winner_rank", "loser_rank"]
    ].copy()
    p2_ranks["p2_id"] = np.maximum(p2_ranks["winner_id"], p2_ranks["loser_id"])
    p2_ranks["p2_rank"] = np.where(
        p2_ranks["p2_id"] == p2_ranks["winner_id"],
        p2_ranks["winner_rank"],
        p2_ranks["loser_rank"],
    )

    features_df = pd.merge(
        features_df, p1_ranks[["match_id", "p1_rank"]], on="match_id"
    )
    features_df = pd.merge(
        features_df, p2_ranks[["match_id", "p2_rank"]], on="match_id"
    )

    features_df["rank_diff"] = features_df["p1_rank"] - features_df["p2_rank"]
    features_df["elo_diff"] = features_df["p1_elo"] - features_df["p2_elo"]
    features_df["surface_elo_diff"] = (
        features_df["p1_surface_elo_pre_match"]
        - features_df["p2_surface_elo_pre_match"]
    )

    features_df = pd.get_dummies(
        features_df, columns=["p1_hand", "p2_hand"], prefix_sep="_"
    )

    # Add player names for the backtester to use
    player_names = df_players[["player_id", "name_first", "name_last"]].copy()
    player_names["player_name"] = (
        player_names["name_first"] + " " + player_names["name_last"]
    )
    player_name_map = player_names[["player_id", "player_name"]].set_index("player_id")
    features_df = features_df.merge(
        player_name_map, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"player_name": "p1_name"})
    features_df = features_df.merge(
        player_name_map, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"player_name": "p2_name"})

    output_path = Path("data/processed/all_advanced_set_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving FINAL SET-LEVEL features to {output_path}...")
    features_df.to_csv(output_path, index=False)

    log_success(
        f"✅ Successfully created FINAL SET-LEVEL feature library at {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)
