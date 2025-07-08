# src/scripts/modeling/train_eval_model.py

import sys
from pathlib import Path
import json
from datetime import datetime
import joblib
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.git_utils import get_git_hash
from src.scripts.utils.logger import log_info, log_success, setup_logging

def train_advanced_model(df: pd.DataFrame, algorithm: str, test_size: float, random_state: int):
    # (The contents of this function remain exactly the same)
    print("Creating a balanced dataset by flipping player perspectives...")
    df_loser = df.copy()
    p1_cols = [col for col in df.columns if col.startswith('p1_')]
    p2_cols = [col for col in df.columns if col.startswith('p2_')]
    swap_map = {**{p1: p2 for p1, p2 in zip(p1_cols, p2_cols)}, 
                **{p2: p1 for p1, p2 in zip(p1_cols, p2_cols)}}
    df_loser = df_loser.rename(columns=swap_map)
    df_loser['winner'] = 0
    df_balanced = pd.concat([df, df_loser], ignore_index=True)
    
    df_balanced = pd.get_dummies(df_balanced, columns=['p1_hand', 'p2_hand'], drop_first=True)
    excluded_cols = ['match_id', 'tourney_date', 'p1_id', 'p2_id', 'winner', 'tourney_name']
    features = [col for col in df_balanced.columns if col not in excluded_cols]
    df_balanced[features] = df_balanced[features].fillna(0)

    print(f"Training model with {len(features)} features.")
    
    X = df_balanced[features]
    y = df_balanced['winner']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training {algorithm.upper()} model on {len(X_train)} samples...")
    model = XGBClassifier(eval_metric="logloss", random_state=random_state, use_label_encoder=False)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, model.predict(X_test), digits=3)
    
    meta = {
        "timestamp": datetime.now().isoformat(), "git_hash": get_git_hash(),
        "model_type": type(model).__name__, "features": features, "algorithm": algorithm,
        "train_rows": len(X_train), "test_rows": len(X_test), "auc": auc,
    }
    return model, report, auc, meta

# --- MODIFIED: Create a main_cli function for importing ---
def main_cli(args):
    """
    Main CLI handler for training the model.
    """
    setup_logging()
    
    print("Consolidating feature files...")
    df = load_dataframes(args.input_glob)

    model, report, auc, meta = train_advanced_model(
        df, algorithm=args.algorithm, test_size=args.test_size, random_state=42
    )

    auc_string = f"{auc:.4f}" if auc is not None else "N/A"
    log_info(f"Evaluation on holdout set (AUC={auc_string}):")
    log_info("\n" + report)

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    log_success(f"Saved model to {args.output_model}")
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    log_success(f"Saved metadata to {output_path.with_suffix('.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced classification model.")
    parser.add_argument("--input_glob", required=True, help="Glob pattern for feature CSVs.")
    parser.add_argument("--output_model", required=True, help="Path to save the trained model file.")
    parser.add_argument("--algorithm", default="xgb", help="Algorithm to use (e.g., 'xgb').")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size.")
    args = parser.parse_args()
    main_cli(args)