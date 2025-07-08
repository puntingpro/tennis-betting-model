# src/scripts/utils/file_utils.py

import glob
import pandas as pd
from typing import List

def load_dataframes(glob_pattern: str, add_source_column: bool = False) -> pd.DataFrame:
    """
    Loads multiple CSVs from a glob pattern into a single DataFrame.
    """
    files = glob.glob(glob_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {glob_pattern}")

    df_list = []
    for f in files:
        df = pd.read_csv(f)
        if add_source_column:
            df['source_file'] = f
        df_list.append(df)
        
    return pd.concat(df_list, ignore_index=True)