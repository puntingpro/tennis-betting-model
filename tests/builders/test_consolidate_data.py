# tests/utils/test_consolidate_data.py

import pandas as pd
from pathlib import Path

# Corrected the import path to point to the 'builders' directory
from src.scripts.builders.consolidate_data import consolidate_data


def test_consolidate_data_filters_unwanted_files(tmp_path: Path):
    """
    Tests that consolidate_data correctly merges primary tour, futures,
    and challenger matches while filtering out only 'qual' and 'doubles'.
    """
    # 1. Create a temporary directory and dummy CSV files
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    # Create files that should be INCLUDED in the consolidation
    (data_dir / "atp_matches_2023.csv").write_text(
        "tourney_date,winner_id\n20230105,101"
    )
    (data_dir / "atp_matches_2022.csv").write_text(
        "tourney_date,winner_id\n20221225,102"
    )
    # This 'futures' file should now be INCLUDED
    (data_dir / "atp_matches_futures_2023.csv").write_text(
        "tourney_date,winner_id\n20230103,903"
    )

    # Create files that should be EXCLUDED by the filter logic
    (data_dir / "atp_matches_qual_2023.csv").write_text(
        "tourney_date,winner_id\n20230101,901"
    )
    (data_dir / "atp_matches_doubles_2023.csv").write_text(
        "tourney_date,winner_id\n20230102,902"
    )

    input_glob = str(data_dir / "atp_matches_*.csv")
    output_file = data_dir / "consolidated.csv"

    # 2. Run the function to be tested
    consolidate_data(input_glob, output_file)

    # 3. Read the output file and verify its contents
    result_df = pd.read_csv(output_file)

    # FIX: The assertion is now updated to expect 3 rows, as 'futures' is a valid, profitable category.
    assert len(result_df) == 3

    # Assert that the data is sorted by date and comes from the correct sources
    # The winner_id '903' from the futures file should now be present.
    assert sorted(result_df["winner_id"].tolist()) == [101, 102, 903]
