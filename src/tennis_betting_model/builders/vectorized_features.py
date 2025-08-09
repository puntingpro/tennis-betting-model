# src/tennis_betting_model/builders/vectorized_features.py

import pandas as pd
from tennis_betting_model.utils.logger import log_info


def build_vectorized_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Builds player features for all historical matches using vectorized pandas operations
    for high performance.
    """
    log_info("Preparing data for vectorization...")

    # Ensure chronological order
    df_matches = df_matches.sort_values("tourney_date").reset_index(drop=True)

    # Create a long-format DataFrame where each row is one player's perspective of a match
    winners = df_matches.copy()
    winners = winners.rename(
        columns={
            "winner_historical_id": "player_id",
            "loser_historical_id": "opponent_id",
        }
    )
    winners["won"] = 1

    losers = df_matches.copy()
    losers = losers.rename(
        columns={
            "loser_historical_id": "player_id",
            "winner_historical_id": "opponent_id",
        }
    )
    losers["won"] = 0

    player_match_df = pd.concat([winners, losers], ignore_index=True)
    player_match_df = player_match_df.sort_values("tourney_date").reset_index(drop=True)

    log_info("Calculating rolling and expanding player statistics...")

    # Calculate point-in-time stats using groupby and expanding/rolling windows
    grouped = player_match_df.groupby("player_id")

    player_stats = pd.DataFrame(
        {
            "win_perc": grouped["won"].expanding().mean().shift(1),
            "form_10": grouped["won"].rolling(window=10, min_periods=1).mean().shift(1),
            "rolling_win_perc_20": grouped["won"]
            .rolling(window=20, min_periods=1)
            .mean()
            .shift(1),
            "rolling_win_perc_50": grouped["won"]
            .rolling(window=50, min_periods=1)
            .mean()
            .shift(1),
        }
    )

    # Surface-specific win percentage
    surface_stats = (
        player_match_df.groupby(["player_id", "surface"])["won"]
        .expanding()
        .mean()
        .shift(1)
    )

    # Fatigue metrics
    player_match_df.set_index("tourney_date", inplace=True)
    fatigue_grouped = player_match_df.groupby("player_id")

    player_stats["matches_last_7_days"] = (
        fatigue_grouped["match_id"].rolling("7D").count().shift(1).values
    )
    player_stats["matches_last_14_days"] = (
        fatigue_grouped["match_id"].rolling("14D").count().shift(1).values
    )
    player_stats["sets_played_last_7_days"] = (
        fatigue_grouped["sets_played"].rolling("7D").sum().shift(1).values
    )
    player_stats["sets_played_last_14_days"] = (
        fatigue_grouped["sets_played"].rolling("14D").sum().shift(1).values
    )

    player_match_df.reset_index(inplace=True)

    # Explicitly align the index of player_stats before concatenation.
    player_stats.index = player_match_df.index
    player_features_df = pd.concat([player_match_df, player_stats], axis=1)

    # Merge surface_stats by making join keys explicit columns
    player_features_df.reset_index(inplace=True)
    surface_stats_df = surface_stats.reset_index(name="surface_win_perc").rename(
        columns={"level_2": "index"}
    )
    player_features_df = player_features_df.merge(
        surface_stats_df, on=["player_id", "surface", "index"], how="left"
    ).drop(columns=["index"])

    # Fill NaNs for players' first matches
    player_features_df.fillna(0, inplace=True)

    log_info("Reconstructing match-wise feature data...")

    # Pivot the data back to a match-wise format
    p1_features = player_features_df.add_prefix("p1_")
    p2_features = player_features_df.add_prefix("p2_")

    # FIX: Perform the merges FIRST, then drop the redundant columns AFTER.
    merged_df = df_matches.merge(
        p1_features,
        left_on=["match_id", "p1_id"],
        right_on=["p1_match_id", "p1_player_id"],
    )

    final_df = merged_df.merge(
        p2_features,
        left_on=["match_id", "p2_id"],
        right_on=["p2_match_id", "p2_player_id"],
    )

    # Define and drop the specific, known redundant columns after all merges are complete
    cols_to_drop = [
        "p1_match_id",
        "p1_player_id",
        "p1_opponent_id",
        "p1_p1_id",
        "p1_p2_id",
        "p2_match_id",
        "p2_player_id",
        "p2_opponent_id",
        "p2_p1_id",
        "p2_p2_id",
    ]
    final_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return final_df
