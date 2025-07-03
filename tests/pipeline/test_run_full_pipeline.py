import pytest
from pathlib import Path
import pandas as pd
from main import main as run_main_cli

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data."""
    return tmp_path

@pytest.fixture
def snapshot_csv(temp_data_dir: Path) -> Path:
    """Create a minimal snapshot CSV for testing."""
    parsed_dir = temp_data_dir / "parsed"
    parsed_dir.mkdir()
    snapshot_data = {
        "market_id": ["1.2345", "1.2345"],
        "timestamp": [1674880000, 1674880000],
        "selection_id": [101, 102],
        "ltp": [2.5, 1.8],
        "volume": [100.0, 150.0],
        "runner_name": ["Player A", "Player B"],
    }
    snapshot_df = pd.DataFrame(snapshot_data)
    snapshot_csv_path = parsed_dir / "snapshot_test.csv"
    snapshot_df.to_csv(snapshot_csv_path, index=False)
    return snapshot_csv_path

@pytest.fixture
def sackmann_csv(temp_data_dir: Path) -> Path:
    """Create a minimal Sackmann data CSV for testing."""
    sackmann_data = {
        "winner_name": ["Player A"],
        "loser_name": ["Player B"],
        "surface": ["Hard"],
        "tourney_date": ["20240115"],
        "match_num": [1],
    }
    sackmann_df = pd.DataFrame(sackmann_data)
    sackmann_csv_path = temp_data_dir / "sackmann_test.csv"
    sackmann_df.to_csv(sackmann_csv_path, index=False)
    return sackmann_csv_path

@pytest.fixture
def pipeline_config_file(temp_data_dir: Path, snapshot_csv: Path, sackmann_csv: Path) -> Path:
    """Create a temporary pipeline run configuration YAML."""
    pipeline_config = f"""
defaults:
  label: test_run
  overwrite: true

stages:
  - build
  - ids

tournaments:
  - label: test_run
    tour: atp
    tournament: test_open
    year: 2024
    snapshots_csv: {snapshot_csv.as_posix()}
    sackmann_csv: {sackmann_csv.as_posix()}
    """
    config_path = temp_data_dir / "test_pipeline_run.yaml"
    config_path.write_text(pipeline_config)
    return config_path


def test_pipeline_run_integration(
    temp_data_dir: Path, pipeline_config_file: Path, monkeypatch
):
    """
    Integration test for the main pipeline orchestrator using pytest fixtures.
    """
    # 1. --- Setup ---
    working_dir = temp_data_dir / "processed"
    working_dir.mkdir()
    
    # 2. --- Execution ---
    cli_args = [
        "main.py",
        "pipeline",
        f"--config={pipeline_config_file.as_posix()}",
        "--batch",
        f"--working_dir={working_dir.as_posix()}",
    ]
    monkeypatch.setattr("sys.argv", cli_args)

    run_main_cli()

    # 3. --- Assertion ---
    expected_build_output = working_dir / "test_run_matches.csv"
    expected_ids_output = working_dir / "test_run_matches_with_ids.csv"

    assert expected_build_output.exists(), "Build stage output file was not created"
    assert expected_ids_output.exists(), "IDs stage output file was not created"

    output_df = pd.read_csv(expected_ids_output)
    assert "selection_id_1" in output_df.columns
    assert "selection_id_2" in output_df.columns
    assert not output_df["selection_id_1"].isnull().all()
    assert not output_df["selection_id_2"].isnull().all()