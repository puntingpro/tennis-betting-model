# tests/builders/test_player_mapper.py

import pandas as pd
import pytest
from pathlib import Path

# The function to be tested
from tennis_betting_model.builders.player_mapper import create_mapping_file


@pytest.fixture
def mock_historical_driven_player_data(tmp_path: Path) -> dict:
    """
    Creates tour-specific historical data to test the historical-driven mapper.
    """
    # 1. Mock Betfair Data (contains a mix of players)
    betfair_data = {
        "selection_id": [101, 201, 301],
        "selection_name": ["Rafael Nadal", "Camila Osorio", "Taylor Fritz"],
        "tourney_name": ["Some Tournament", "Another Tournament", "A Third Event"],
    }
    df_betfair = pd.DataFrame(betfair_data)
    betfair_csv_path = tmp_path / "betfair_raw_odds.csv"
    df_betfair.to_csv(betfair_csv_path, index=False)

    # 2. Mock Historical ATP Data
    atp_dir = tmp_path / "tennis_atp"
    atp_dir.mkdir()
    atp_historical_data = {
        "winner_id": [1, 2],
        "winner_name": [
            "Rafael Nadal",
            "Camilo Osorio",
        ],  # Male player with similar name
        "loser_id": [3, 4],
        "loser_name": ["R. Federer", "Taylor Fritz"],
    }
    df_atp_historical = pd.DataFrame(atp_historical_data)
    atp_csv_path = atp_dir / "atp_matches_2023.csv"
    df_atp_historical.to_csv(atp_csv_path, index=False)

    # 3. Mock Historical WTA Data
    wta_dir = tmp_path / "tennis_wta"
    wta_dir.mkdir()
    wta_historical_data = {
        "winner_id": [5],
        "winner_name": ["Camila Osorio"],  # Female player
        "loser_id": [6],
        "loser_name": ["I. Swiatek"],
    }
    df_wta_historical = pd.DataFrame(wta_historical_data)
    wta_csv_path = wta_dir / "wta_matches_2023.csv"
    df_wta_historical.to_csv(wta_csv_path, index=False)

    # 4. Mock Config
    mock_config = {
        "data_paths": {
            "betfair_raw_odds": str(betfair_csv_path),
            "raw_data_dir": str(tmp_path),
            "player_map": str(tmp_path / "player_mapping.csv"),
        },
        "mapping_params": {"confidence_threshold": 85},
    }
    return mock_config


def test_create_mapping_file_uses_historical_data_for_tour(
    mock_historical_driven_player_data,
):
    """
    Tests that the refactored mapper correctly uses historical data to determine
    a player's tour, avoiding cross-gender mapping errors.
    """
    create_mapping_file(mock_historical_driven_player_data)

    output_path = Path(mock_historical_driven_player_data["data_paths"]["player_map"])
    assert output_path.exists()
    result_df = pd.read_csv(output_path)

    results = result_df.set_index("betfair_name").to_dict("index")

    # Check Camila Osorio (WTA) was mapped to the WTA historical ID (5), not the ATP one (2)
    osorio_result = results.get("Camila Osorio")
    assert osorio_result is not None
    assert osorio_result["historical_id"] == 5  # This is the WTA ID
    assert osorio_result["matched_name"] == "Camila Osorio"

    # Check Rafael Nadal (ATP) was mapped to the correct ID (1)
    nadal_result = results.get("Rafael Nadal")
    assert nadal_result is not None
    assert nadal_result["historical_id"] == 1  # This is the ATP ID

    # Check Taylor Fritz (ATP) was mapped to the correct ID (4)
    fritz_result = results.get("Taylor Fritz")
    assert fritz_result is not None
    assert fritz_result["historical_id"] == 4  # This is the ATP ID
