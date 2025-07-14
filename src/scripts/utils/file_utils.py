# src/scripts/utils/file_utils.py

import glob
import pandas as pd
from typing import List


def load_dataframes(glob_pattern: str, add_source_column: bool = False) -> pd.DataFrame:
    """
    Loads and concatenates multiple CSV files from a glob pattern into a single DataFrame.

    Args:
        glob_pattern (str): The glob pattern to match CSV files (e.g., 'data/*.csv').
        add_source_column (bool): If True, adds a 'source_file' column to indicate
                                  the origin of each row.

    Returns:
        pd.DataFrame: A single DataFrame containing data from all matched files.

    Raises:
        FileNotFoundError: If no files are found matching the provided glob pattern.
    """
    files = glob.glob(glob_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {glob_pattern}")

    df_list: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        if add_source_column:
            df["source_file"] = f
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)
