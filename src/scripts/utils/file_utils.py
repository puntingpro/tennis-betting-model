import glob
from pathlib import Path
from typing import List

import pandas as pd
from .logger import log_error, log_info, log_success

def get_files_from_glob(file_glob: str) -> List[str]:
    """
    Returns a list of files matching the provided glob pattern.
    """
    if "*" not in file_glob:
        p = Path(file_glob)
        if not p.exists():
            raise FileNotFoundError(f"Specified file not found: {file_glob}")
        return [file_glob]

    files = glob.glob(file_glob, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching glob pattern: {file_glob}")
    return files


def load_dataframes(file_glob: str, add_source_column: bool = False) -> pd.DataFrame:
    """
    Loads one or more CSVs from a glob pattern into a single pandas DataFrame.
    """
    files = get_files_from_glob(file_glob)
    log_info(f"Found {len(files)} files matching '{file_glob}'.")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if add_source_column:
                df['source_file'] = f
            dfs.append(df)
        except pd.errors.EmptyDataError:
            log_info(f"Skipping empty file: {f}")
        except Exception as e:
            log_error(f"Error loading {f}: {e}")
            
    if not dfs:
        raise ValueError(f"No data loaded from files matching glob pattern: {file_glob}")

    combined_df = pd.concat(dfs, ignore_index=True)
    log_info(f"Successfully loaded and combined {len(files)} files into a DataFrame with {len(combined_df)} rows.")
    return combined_df

def write_csv(df: pd.DataFrame, path: str, overwrite: bool = True):
    """
    Writes a DataFrame to a CSV, creating parent directories if needed.
    """
    p = Path(path)
    if not overwrite and p.exists():
        log_info(f"File already exists and overwrite is False, skipping: {path}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    log_success(f"Successfully saved data to {p}")