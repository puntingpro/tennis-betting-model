# src/scripts/utils/schema.py

import pandas as pd

# Define canonical schemas for different stages of the pipeline
# This helps enforce consistency.
SCHEMAS = {
    "predictions": [
        "match_id", "player_1", "player_2", "predicted_prob", "odds"
    ],
    "value_bets": [
        "match_id", "player_1", "player_2", "predicted_prob", "odds", 
        "expected_value", "kelly_fraction", "confidence_score"
    ],
    "simulations": [
        "match_id", "player_name", "odds", "predicted_prob", "winner",
        "expected_value", "kelly_fraction", "stake", "profit", "bankroll", "strategy"
    ]
}

def enforce_schema(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Enforces a specific schema on a DataFrame. Adds missing columns
    with pd.NA and reorders columns to match the schema.
    """
    if schema_name not in SCHEMAS:
        raise ValueError(f"Schema '{schema_name}' not defined.")
        
    schema_cols = SCHEMAS[schema_name]
    
    # Add missing columns
    for col in schema_cols:
        if col not in df.columns:
            df[col] = pd.NA
            
    # Return DataFrame with columns in the specified order
    return df[schema_cols]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names to lowercase and replaces spaces with underscores.
    """
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def patch_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the 'winner' column, ensuring it's numeric.
    It checks for a 'result' column and converts it if 'winner' is missing.
    """
    if 'winner' not in df.columns and 'result' in df.columns:
        df['winner'] = pd.to_numeric(df['result'], errors='coerce')
    elif 'winner' in df.columns:
        df['winner'] = pd.to_numeric(df['winner'], errors='coerce')
        
    return df