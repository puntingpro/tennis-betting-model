import pandas as pd
import pytest
from pathlib import Path

from tennis_betting_model.builders.player_mapper import run_create_mapping_file
from tennis_betting_model.utils.config_schema import DataPaths, MappingParams


@pytest.fixture
def mock_historical_driven_player_data(tmp_path: Path) -> dict:
    """
    Creates tour-specific historical data to test the historical-driven mapper.
    """
    betfair_data = {
        "selection_id": [101, 201, 301],
        "selection_name": ["Rafael Nadal", "Camila Osorio", "Taylor Fritz"],
        "tourney_name": ["Some Tournament", "Another Tournament", "A Third Event"],
    }
    df_betfair = pd.DataFrame(betfair_data)
    betfair_csv_path = tmp_path / "betfair_raw_odds.csv"
    df_betfair.to_csv(betfair_csv_path, index=False)

    atp_dir = tmp_path / "tennis_atp"
    atp_dir.mkdir()
    atp_historical_data = {
        "winner_id": [1, 2],
        "winner_name": ["Rafael Nadal", "Camilo Osorio"],
        "loser_id": [3, 4],
        "loser_name": ["R. Federer", "Taylor Fritz"],
    }
    df_atp_historical = pd.DataFrame(atp_historical_data)
    atp_csv_path = atp_dir / "atp_matches_2023.csv"
    df_atp_historical.to_csv(atp_csv_path, index=False)

    wta_dir = tmp_path / "tennis_wta"
    wta_dir.mkdir()
    wta_historical_data = {
        "winner_id": [5],
        "winner_name": ["Camila Osorio"],
        "loser_id": [6],
        "loser_name": ["I. Swiatek"],
    }
    df_wta_historical = pd.DataFrame(wta_historical_data)
    wta_csv_path = wta_dir / "wta_matches_2023.csv"
    df_wta_historical.to_csv(wta_csv_path, index=False)

    # Add all required paths for DataPaths model validation, even if empty
    mock_config = {
        "data_paths": {
            "betfair_raw_odds": str(betfair_csv_path),
            "raw_data_dir": str(tmp_path),
            "player_map": str(tmp_path / "player_mapping.csv"),
            "processed_data_dir": str(tmp_path / "processed"),
            "plot_dir": str(tmp_path / "plots"),
            "raw_players": str(tmp_path / "players.csv"),
            "consolidated_rankings": str(tmp_path / "rankings.csv"),
            "betfair_match_log": str(tmp_path / "match_log.csv"),
            "elo_ratings": str(tmp_path / "elo.csv"),
            "consolidated_features": str(tmp_path / "features.csv"),
            "backtest_market_data": str(tmp_path / "backtest_market.csv"),
            "model": str(tmp_path / "model.joblib"),
            "backtest_results": str(tmp_path / "results.csv"),
            "tournament_summary": str(tmp_path / "summary.csv"),
            "processed_bets_log": str(tmp_path / "bets.db"),
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
    config = mock_historical_driven_player_data
    # Instantiate the Pydantic models from the mock config
    data_paths = DataPaths(**config["data_paths"])
    mapping_params = MappingParams(**config["mapping_params"])

    # Call the function with the correct arguments
    run_create_mapping_file(data_paths, mapping_params)

    output_path = Path(config["data_paths"]["player_map"])
    assert output_path.exists()
    result_df = pd.read_csv(output_path)

    results = result_df.set_index("betfair_name").to_dict("index")

    osorio_result = results.get("Camila Osorio")
    assert osorio_result is not None
    assert osorio_result["historical_id"] == 5

    nadal_result = results.get("Rafael Nadal")
    assert nadal_result is not None
    assert nadal_result["historical_id"] == 1

    fritz_result = results.get("Taylor Fritz")
    assert fritz_result is not None
    assert fritz_result["historical_id"] == 4
