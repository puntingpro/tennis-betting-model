# src/scripts/analysis/backtest_strategy.py
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import argparse
import numpy as np

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.logger import log_info, log_success, setup_logging
from src.scripts.utils.betting_math import add_ev_and_kelly
from src.scripts.utils.config import load_config

# --- Define constants for a more realistic backtest ---
MAX_ODDS = 50.0
BOOKMAKER_MARGIN = 1.05 # Represents a 5% margin

def run_backtest(model_path: str, features_csv: str, output_csv: str, ev_threshold: float):
    log_info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    model_features = model.feature_names_in_

    log_info(f"Loading historical features from {features_csv}...")
    df = pd.read_csv(features_csv, low_memory=False)

    # --- Prepare Data for Prediction ---
    df_predict = df.copy()

    for col in model_features:
        if col in df_predict.columns:
            df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')

    # Ensure all dummy columns from training are present
    df_predict = pd.get_dummies(df_predict, columns=['p1_hand', 'p2_hand'], drop_first=True)

    final_model_features = model.feature_names_in_
    for col in final_model_features:
        if col not in df_predict.columns:
            df_predict[col] = 0.0 # Add missing columns with 0

    # Ensure order and columns match the model
    df_predict = df_predict.reindex(columns=final_model_features, fill_value=0)

    X_historical = df_predict[final_model_features]

    # --- Make Predictions ---
    log_info("Making predictions on historical data...")
    df['p1_predicted_prob'] = model.predict_proba(X_historical)[:, 1]
    df['p2_predicted_prob'] = 1 - df['p1_predicted_prob']

    # --- Simulate Odds and Find Value ---
    log_info("Simulating odds and finding value for both players in each match...")
    df['p1_win_perc'] = pd.to_numeric(df['p1_win_perc'], errors='coerce').fillna(0)
    df['p2_win_perc'] = pd.to_numeric(df['p2_win_perc'], errors='coerce').fillna(0)

    total_perc = df['p1_win_perc'] + df['p2_win_perc']
    df['p1_true_prob'] = np.where(total_perc > 0, df['p1_win_perc'] / total_perc, 0.5)
    df['p2_true_prob'] = np.where(total_perc > 0, df['p2_win_perc'] / total_perc, 0.5)

    # --- MODIFIED SECTION: Add bookmaker margin and cap the simulated odds ---
    df['p1_odds'] = np.where(df['p1_true_prob'] > 0, (1 / df['p1_true_prob']) / BOOKMAKER_MARGIN, MAX_ODDS)
    df['p1_odds'] = df['p1_odds'].clip(upper=MAX_ODDS)

    df['p2_odds'] = np.where(df['p2_true_prob'] > 0, (1 / df['p2_true_prob']) / BOOKMAKER_MARGIN, MAX_ODDS)
    df['p2_odds'] = df['p2_odds'].clip(upper=MAX_ODDS)
    # --- END OF MODIFIED SECTION ---

    base_cols = ['match_id', 'tourney_name']

    bets_p1 = df[base_cols].copy()
    bets_p1['odds'] = df['p1_odds']
    bets_p1['predicted_prob'] = df['p1_predicted_prob']
    bets_p1['winner'] = 1

    bets_p2 = df[base_cols].copy()
    bets_p2['odds'] = df['p2_odds']
    bets_p2['predicted_prob'] = df['p2_predicted_prob']
    bets_p2['winner'] = 0

    bets_p1 = add_ev_and_kelly(bets_p1, inplace=False)
    value_bets_p1 = bets_p1[bets_p1['expected_value'] > ev_threshold]

    bets_p2 = add_ev_and_kelly(bets_p2, inplace=False)
    value_bets_p2 = bets_p2[bets_p2['expected_value'] > ev_threshold]

    final_value_bets = pd.concat([value_bets_p1, value_bets_p2], ignore_index=True)

    log_success(f"Found {len(final_value_bets)} total historical value bets.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_value_bets.to_csv(output_path, index=False)
    log_success(f"Saved final backtest results to {output_path}")

def main(args):
    """
    Main function for backtesting, driven by the config file.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']
    betting_params = config['betting']

    run_backtest(
        model_path=paths['model'],
        features_csv=paths['consolidated_features'],
        output_csv=paths['backtest_results'],
        ev_threshold=betting_params['ev_threshold']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a historical backtest using a trained model.")
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    main(args)