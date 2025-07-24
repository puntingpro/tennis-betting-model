# src/scripts/builders/build_player_features.py

from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import glob

from scripts.utils.config import load_config
from scripts.utils.logger import setup_logging, log_info, log_success
from scripts.utils.constants import DEFAULT_PLAYER_RANK, ELO_INITIAL_RATING


def load_charting_data(charting_path: Path) -> pd.DataFrame:
    """Loads and combines all point-by-point charting data."""
    log_info("Loading point-by-point charting data...")
    try:
        m_matches_df = pd.read_csv(
            charting_path / "charting-m-matches.csv", encoding="latin-1"
        )
        m_matches_df.rename(
            columns={"Player 1": "p1_charting_name", "Player 2": "p2_charting_name"},
            inplace=True,
        )

        w_matches_df = pd.read_csv(
            charting_path / "charting-w-matches.csv", encoding="latin-1"
        )
        w_matches_df.rename(
            columns={"Player 1": "p1_charting_name", "Player 2": "p2_charting_name"},
            inplace=True,
        )

        all_matches = pd.concat([m_matches_df, w_matches_df], ignore_index=True)
        all_matches["Date"] = pd.to_datetime(all_matches["Date"], errors="coerce")

        points_files = glob.glob(str(charting_path / "charting-*-points-*.csv"))
        if not points_files:
            raise FileNotFoundError(
                "No point-by-point CSV files found with the pattern 'charting-*-points-*.csv'"
            )

        points_dfs = [
            pd.read_csv(f, encoding="latin-1", low_memory=False) for f in points_files
        ]
        all_points = pd.concat(points_dfs, ignore_index=True)

        charting_df = pd.merge(
            all_points,
            all_matches[["match_id", "Date", "p1_charting_name", "p2_charting_name"]],
            on="match_id",
            how="left",
        )
        return charting_df
    except FileNotFoundError as e:
        log_info(
            f"Could not find charting files, skipping point-level features. Error: {e}"
        )
        return pd.DataFrame()


def calculate_point_level_stats(charting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rolling point-level stats by identifying break points from the point score.
    """
    if charting_df.empty:
        return pd.DataFrame()

    log_info("Calculating point-level stats with anti-leakage logic...")
    charting_df.sort_values(by=["Date", "match_id", "Pt"], inplace=True)

    player_stats_cache = defaultdict(
        lambda: {"bp_faced": 0, "bp_saved": 0, "bp_opportunities": 0, "bp_converted": 0}
    )

    match_point_features = []

    for match_id, match_points in tqdm(
        charting_df.groupby("match_id"), desc="Processing Point-Level Stats"
    ):
        p1_name = match_points["p1_charting_name"].iloc[0]
        p2_name = match_points["p2_charting_name"].iloc[0]

        p1_stats = player_stats_cache[p1_name]
        p2_stats = player_stats_cache[p2_name]

        match_point_features.append(
            {
                "match_id": match_id,
                "p1_bp_save_perc": (p1_stats["bp_saved"] / p1_stats["bp_faced"])
                if p1_stats["bp_faced"] > 0
                else 0,
                "p1_bp_convert_perc": (
                    p1_stats["bp_converted"] / p1_stats["bp_opportunities"]
                )
                if p1_stats["bp_opportunities"] > 0
                else 0,
                "p2_bp_save_perc": (p2_stats["bp_saved"] / p2_stats["bp_faced"])
                if p2_stats["bp_faced"] > 0
                else 0,
                "p2_bp_convert_perc": (
                    p2_stats["bp_converted"] / p2_stats["bp_opportunities"]
                )
                if p2_stats["bp_opportunities"] > 0
                else 0,
            }
        )

        for row in match_points.itertuples():
            server = row.Svr
            point_winner = row.PtWinner
            score = str(row.Pts)

            is_bp = False
            if "AD" in score and server == 2:
                is_bp = True
            elif "40" in score and "AD" not in score:
                try:
                    p1_score, p2_score = map(int, score.split("-"))
                    if server == 1 and p2_score == 40:
                        is_bp = True
                    elif server == 2 and p1_score == 40:
                        is_bp = True
                except ValueError:
                    continue

            if is_bp:
                if server == 1:
                    p1_stats["bp_faced"] += 1
                    p2_stats["bp_opportunities"] += 1
                    if point_winner == 1:
                        p1_stats["bp_saved"] += 1
                    else:
                        p2_stats["bp_converted"] += 1
                else:
                    p2_stats["bp_faced"] += 1
                    p1_stats["bp_opportunities"] += 1
                    if point_winner == 2:
                        p2_stats["bp_saved"] += 1
                    else:
                        p1_stats["bp_converted"] += 1

    return pd.DataFrame(match_point_features)


def parse_score_to_sets(df: pd.DataFrame) -> pd.DataFrame:
    log_info("Parsing match scores to determine set winners...")
    sets_data = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Parsing Scores"):
        try:
            scores = str(row.score).split(" ")
            for i, set_score in enumerate(scores):
                if "-" in set_score and "RET" not in set_score:
                    p1_games_str, p2_games_str = set_score.split("-")
                    p1_games = int(p1_games_str.strip("()[]"))
                    p2_games = int(p2_games_str.strip("()[]"))
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
    log_info("Calculating surface-specific Elo ratings...")
    surface_elos = defaultdict(lambda: defaultdict(lambda: ELO_INITIAL_RATING))
    elo_results = []
    for row in tqdm(
        df_matches.itertuples(), total=len(df_matches), desc="Elo Calculation"
    ):
        winner_id, loser_id, surface = row.winner_id, row.loser_id, row.surface
        winner_elo, loser_elo = (
            surface_elos[winner_id][surface],
            surface_elos[loser_id][surface],
        )
        expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        k = 32
        new_winner_elo, new_loser_elo = winner_elo + k * (
            1 - expected_win
        ), loser_elo + k * (0 - (1 - expected_win))
        surface_elos[winner_id][surface], surface_elos[loser_id][surface] = (
            new_winner_elo,
            new_loser_elo,
        )
        p1_id, p2_id = min(winner_id, loser_id), max(winner_id, loser_id)
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
    log_info("Calculating player stats with anti-leakage logic...")
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

    for row in tqdm(
        df_matches_with_ranks.itertuples(),
        total=len(df_matches_with_ranks),
        desc="Processing matches",
    ):
        winner_id, loser_id, date, surface = (
            row.winner_id,
            row.loser_id,
            row.tourney_date,
            row.surface,
        )
        winner_rank, loser_rank = row.winner_rank, row.loser_rank

        winner_stats = player_stats_cache[winner_id]
        loser_stats = player_stats_cache[loser_id]

        for player_id, stats, opponent_rank in [
            (winner_id, winner_stats, loser_rank),
            (loser_id, loser_stats, winner_rank),
        ]:
            match_features.append(
                {
                    "match_id": row.match_id,
                    "player_id": player_id,
                    "win_perc": (stats["wins"] / stats["matches_played"])
                    if stats["matches_played"] > 0
                    else 0,
                    "surface_win_perc": (
                        stats["surface_wins"][surface]
                        / stats["surface_matches"][surface]
                    )
                    if stats["surface_matches"][surface] > 0
                    else 0,
                    "form_last_10": np.mean(stats["form"]) if stats["form"] else 0,
                    "rest_days": (date - stats["last_match_date"]).days
                    if pd.notna(stats["last_match_date"])
                    else 30,
                    "sets_played_last_7d": sum(
                        c for d, c in stats["recent_sets"] if (date - d).days <= 7
                    ),
                    "avg_opponent_rank_last_10": np.mean(stats["opponent_ranks"])
                    if stats["opponent_ranks"]
                    else DEFAULT_PLAYER_RANK,
                }
            )

        winner_stats.update(
            {
                "matches_played": winner_stats["matches_played"] + 1,
                "wins": winner_stats["wins"] + 1,
                "last_match_date": date,
            }
        )
        winner_stats["surface_matches"][surface] += 1
        winner_stats["surface_wins"][surface] += 1
        winner_stats["form"] = (winner_stats["form"] + [1])[-10:]
        winner_stats["opponent_ranks"] = (
            winner_stats["opponent_ranks"] + [loser_rank]
        )[-10:]
        try:
            sets_played = len(str(row.score).split())
            winner_stats["recent_sets"] = [
                s for s in winner_stats["recent_sets"] if (date - s[0]).days <= 30
            ] + [(date, sets_played)]
        except:
            pass

        loser_stats.update(
            {
                "matches_played": loser_stats["matches_played"] + 1,
                "wins": loser_stats["wins"],
                "last_match_date": date,
            }
        )
        loser_stats["surface_matches"][surface] += 1
        loser_stats["form"] = (loser_stats["form"] + [0])[-10:]
        loser_stats["opponent_ranks"] = (loser_stats["opponent_ranks"] + [winner_rank])[
            -10:
        ]
        try:
            sets_played = len(str(row.score).split())
            loser_stats["recent_sets"] = [
                s for s in loser_stats["recent_sets"] if (date - s[0]).days <= 30
            ] + [(date, sets_played)]
        except:
            pass

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
    charting_path = Path(paths["raw_data_dir"]) / "charting"
    charting_df = load_charting_data(charting_path)
    point_level_stats_df = calculate_point_level_stats(charting_df)
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
    df_matches["winner_id"], df_matches["loser_id"] = df_matches["winner_id"].astype(
        "int64"
    ), df_matches["loser_id"].astype("int64")
    df_rankings.dropna(subset=["player"], inplace=True)
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
        df_matches[["tourney_date", "winner_id"]],
        df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="winner_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "winner_rank"})
    loser_ranks = pd.merge_asof(
        df_matches[["tourney_date", "loser_id"]],
        df_rankings_sorted[["ranking_date", "player", "rank"]],
        left_on="tourney_date",
        right_on="ranking_date",
        left_by="loser_id",
        right_by="player",
        direction="backward",
    ).rename(columns={"rank": "loser_rank"})
    df_matches["winner_rank"], df_matches["loser_rank"] = winner_ranks[
        "winner_rank"
    ].fillna(DEFAULT_PLAYER_RANK), loser_ranks["loser_rank"].fillna(DEFAULT_PLAYER_RANK)
    player_stats_df = calculate_player_stats(df_matches)
    df_surface_elo = calculate_surface_elo(df_matches)
    df_sets = parse_score_to_sets(df_matches)
    log_info("Assembling final feature set...")
    features_df = df_sets[["match_id", "set_num", "set_winner_id"]]
    match_players = df_matches[["match_id", "winner_id", "loser_id"]].copy()
    match_players["p1_id"], match_players["p2_id"] = np.minimum(
        match_players["winner_id"], match_players["loser_id"]
    ), np.maximum(match_players["winner_id"], match_players["loser_id"])
    features_df = pd.merge(
        features_df, match_players[["match_id", "p1_id", "p2_id"]], on="match_id"
    )
    features_df["winner"] = (
        features_df["p1_id"] == features_df["set_winner_id"]
    ).astype(int)
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
    )
    p2_stats = player_stats_df.rename(
        columns=lambda c: f"p2_{c}" if c not in ["match_id", "player_id"] else c
    )
    features_df = pd.merge(
        features_df,
        p2_stats,
        left_on=["match_id", "p2_id"],
        right_on=["match_id", "player_id"],
    )
    player_info = df_players[["player_id", "hand", "height"]].set_index("player_id")
    features_df = features_df.merge(
        player_info, left_on="p1_id", right_index=True, how="left"
    ).rename(columns={"hand": "p1_hand", "height": "p1_height"})
    features_df = features_df.merge(
        player_info, left_on="p2_id", right_index=True, how="left"
    ).rename(columns={"hand": "p2_hand", "height": "p2_height"})

    ranks = df_matches[
        ["match_id", "winner_id", "loser_id", "winner_rank", "loser_rank"]
    ].copy()
    ranks["p1_id"] = np.minimum(ranks["winner_id"], ranks["loser_id"])
    ranks["p2_id"] = np.maximum(ranks["winner_id"], ranks["loser_id"])
    ranks["p1_rank"] = np.where(
        ranks["p1_id"] == ranks["winner_id"], ranks["winner_rank"], ranks["loser_rank"]
    )
    ranks["p2_rank"] = np.where(
        ranks["p2_id"] == ranks["winner_id"], ranks["winner_rank"], ranks["loser_rank"]
    )
    features_df = pd.merge(
        features_df, ranks[["match_id", "p1_rank", "p2_rank"]], on="match_id"
    )

    (
        features_df["rank_diff"],
        features_df["elo_diff"],
        features_df["surface_elo_diff"],
    ) = (
        features_df["p1_rank"] - features_df["p2_rank"],
        features_df["p1_elo"] - features_df["p2_elo"],
        features_df["p1_surface_elo_pre_match"]
        - features_df["p2_surface_elo_pre_match"],
    )
    features_df = pd.get_dummies(
        features_df, columns=["p1_hand", "p2_hand"], prefix_sep="_"
    )
    player_names = df_players[["player_id", "name_first", "name_last"]].copy()
    player_names["player_name"] = (
        player_names["name_first"] + " " + player_names["name_last"]
    )

    # --- FINAL FIX: Drop the duplicate player_id column after each merge ---
    features_df = (
        features_df.merge(
            player_names[["player_id", "player_name"]],
            left_on="p1_id",
            right_on="player_id",
        )
        .rename(columns={"player_name": "p1_name"})
        .drop("player_id", axis=1)
    )
    features_df = (
        features_df.merge(
            player_names[["player_id", "player_name"]],
            left_on="p2_id",
            right_on="player_id",
        )
        .rename(columns={"player_name": "p2_name"})
        .drop("player_id", axis=1)
    )

    if not point_level_stats_df.empty:
        features_df = pd.merge(
            features_df, point_level_stats_df, on=["match_id"], how="left"
        )
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
