from typing import Dict, Optional


def build_market_runner_map(df) -> Dict[str, Dict[str, int]]:
    """
    Builds a nested dictionary mapping market_id -> runner_name -> selection_id.
    """
    market_map: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        market_id = str(row["market_id"])
        runner_name = str(row["runner_name"])
        selection_id = int(row["selection_id"])
        if market_id not in market_map:
            market_map[market_id] = {}
        market_map[market_id][runner_name] = selection_id
    return market_map


def match_player_to_selection_id(
    market_map: Dict[str, Dict[str, int]], market_id: str, player_name: str
) -> Optional[int]:
    """
    Finds the selection_id for a player in a given market.
    """
    market_id_str = str(market_id)
    if market_id_str in market_map:
        return market_map[market_id_str].get(player_name)
    return None