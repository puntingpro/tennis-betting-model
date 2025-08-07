import pandas as pd
from typing import cast


def get_h2h_stats_optimized(
    h2h_df: pd.DataFrame, p1_id: int, p2_id: int, match_date: pd.Timestamp
) -> tuple[int, int]:
    """
    Calculates point-in-time Head-to-Head (H2H) stats using a pre-indexed DataFrame.
    """
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

    player1 = min(p1_id, p2_id)
    player2 = max(p1_id, p2_id)

    h2h_matches: pd.DataFrame = pd.DataFrame()
    try:
        h2h_matches_lookup = h2h_df.loc[(player1, player2)]

        if isinstance(h2h_matches_lookup, pd.Series):
            h2h_matches = h2h_matches_lookup.to_frame().T
        else:
            h2h_matches = cast(pd.DataFrame, h2h_matches_lookup)

    except KeyError:
        return 0, 0

    past_matches = h2h_matches[h2h_matches["tourney_date"] < match_date]

    if past_matches.empty:
        return 0, 0

    p1_wins = (past_matches["winner_historical_id"] == p1_id).sum()
    p2_wins = (past_matches["winner_historical_id"] == p2_id).sum()

    return int(p1_wins), int(p2_wins)


def get_player_stats_optimized(
    player_match_df: pd.DataFrame,
    player_id: int,
    surface: str,
    match_date: pd.Timestamp,
) -> tuple[float, float, float, int, int]:
    """
    Calculates point-in-time stats for a player using a pre-indexed, player-centric DataFrame.
    """
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

    all_player_matches: pd.DataFrame = pd.DataFrame()
    try:
        player_matches_lookup = player_match_df.loc[player_id]

        if isinstance(player_matches_lookup, pd.Series):
            all_player_matches = player_matches_lookup.to_frame().T
        else:
            all_player_matches = cast(pd.DataFrame, player_matches_lookup)

    except KeyError:
        return 0.0, 0.0, 0.0, 0, 0

    player_matches = all_player_matches[all_player_matches["tourney_date"] < match_date]

    if player_matches.empty:
        return 0.0, 0.0, 0.0, 0, 0

    win_perc = float(player_matches["won"].mean())

    surface_matches = player_matches[player_matches["surface"] == surface]
    surface_win_perc = (
        float(surface_matches["won"].mean()) if not surface_matches.empty else 0.0
    )

    last_10 = player_matches.tail(10)
    form_last_10 = float(last_10["won"].mean()) if not last_10.empty else 0.0

    # --- FIX: Add 'type: ignore' to suppress the pandas-specific mypy error ---
    time_since_matches = match_date - player_matches["tourney_date"]  # type: ignore[operator]

    matches_last_14_days = (time_since_matches.dt.days <= 14).sum()
    matches_last_7_days = (time_since_matches.dt.days <= 7).sum()

    return (
        win_perc,
        surface_win_perc,
        form_last_10,
        int(matches_last_7_days),
        int(matches_last_14_days),
    )
