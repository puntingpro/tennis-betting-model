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
from src.scripts.utils.constants import DEFAULT_EV_THRESHOLD

def run_backtest(model_path: str, features_csv: str, output_csv: str):
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

    df_predict = pd.get_dummies(df_predict, columns=['p1_hand', 'p2_hand'], drop_first=True)
    
    final_model_features = model.feature_names_in_
    for col in final_model_features:
        if col not in df_predict.columns:
            df_predict[col] = 0.0
    df_predict = df_predict[final_model_features].fillna(0)
    
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
    
    df['p1_odds'] = np.where(df['p1_true_prob'] > 0, 1 / df['p1_true_prob'], 200.0)
    df['p2_odds'] = np.where(df['p2_true_prob'] > 0, 1 / df['p2_true_prob'], 200.0)
    
    # --- MODIFICATION: Ensure tourney_name is preserved ---
    base_cols = ['match_id', 'tourney_date', 'tourney_name']
    
    bets_p1 = df[base_cols].copy()
    bets_p1['odds'] = df['p1_odds']
    bets_p1['predicted_prob'] = df['p1_predicted_prob']
    bets_p1['winner'] = 1
    
    bets_p2 = df[base_cols].copy()
    bets_p2['odds'] = df['p2_odds']
    bets_p2['predicted_prob'] = df['p2_predicted_prob']
    bets_p2['winner'] = 0
    # --- END MODIFICATION ---

    bets_p1 = add_ev_and_kelly(bets_p1, inplace=False)
    value_bets_p1 = bets_p1[bets_p1['expected_value'] > DEFAULT_EV_THRESHOLD]

    bets_p2 = add_ev_and_kelly(bets_p2, inplace=False)
    value_bets_p2 = bets_p2[bets_p2['expected_value'] > DEFAULT_EV_THRESHOLD]

    final_value_bets = pd.concat([value_bets_p1, value_bets_p2], ignore_index=True)
    
    log_success(f"Found {len(final_value_bets)} total historical value bets.")
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_value_bets.to_csv(output_path, index=False)
    log_success(f"Saved final backtest results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run a historical backtest using a trained model.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    setup_logging()
    run_backtest(args.model_path, args.features_csv, args.output_csv)

if __name__ == "__main__":
    main()