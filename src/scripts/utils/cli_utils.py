"""
Common CLI utilities for file and DataFrame handling.
"""

import os


def assert_file_exists(path: str, description: str) -> None:
    """
    Ensure that the given file path exists, otherwise raise FileNotFoundError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def assert_columns_exist(df, columns, context: str = "") -> None:
    """
    Check that all specified columns exist in the DataFrame;
    raise ValueError otherwise.
    """
    missing = set(columns) - set(df.columns)
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}Missing columns: {sorted(missing)}")