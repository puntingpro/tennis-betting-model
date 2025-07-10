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
import optuna
# 1. --- IMPORT THE PRUNING CALLBACK ---
from optuna.integration import XGBoostPruningCallback


# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.git_utils import get_git_hash
from src.scripts.utils.logger import log_info, log_success, setup_logging
from src.scripts.utils.config import load_config

def train_advanced_model(df: pd.DataFrame, test_size: float, random_state: int, n_trials: int):
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
    excluded_cols = ['match_id', 'tourney_date', 'p1_id', 'p2_id', 'winner', 'tourney_name', 'surface']
    features = [col for col in df_balanced.columns if col not in excluded_cols]
    df_balanced[features] = df_balanced[features].fillna(0)

    print(f"Training model with {len(features)} features.")

    X = df_balanced[features]
    y = df_balanced['winner']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': random_state
        }
        model = XGBClassifier(**param, use_label_encoder=False)
        
        # 3. --- CREATE THE PRUNING CALLBACK ---
        # This tells Optuna how to monitor the model's performance.
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False,
            # 4. --- ADD THE CALLBACK TO THE FIT METHOD ---
            callbacks=[pruning_callback]
        )
        
        y_prob = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_prob)

    # 2. --- ADD A PRUNER TO THE STUDY ---
    # This tells Optuna to use the median of past trials to stop unpromising new ones.
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1) # Using n_jobs=-1 from before

    log_info(f"Best trial: {study.best_trial.value}")
    log_info(f"Best params: {study.best_params}")

    model = XGBClassifier(**study.best_params, use_label_encoder=False, random_state=random_state)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, model.predict(X_test), digits=3)

    meta = {
        "timestamp": datetime.now().isoformat(), "git_hash": get_git_hash(),
        "model_type": type(model).__name__, "features": features, "algorithm": "xgb",
        "train_rows": len(X_train), "test_rows": len(X_test), "auc": auc,
        "best_params": study.best_params
    }
    return model, report, auc, meta

# --- MODIFIED: Create a main_cli function for importing ---
def main_cli(args):
    """
    Main CLI handler for training the model.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']
    params = config['model_params']

    print("Consolidating feature files...")
    df = load_dataframes(paths['consolidated_features'])

    model, report, auc, meta = train_advanced_model(
        df,
        test_size=params['test_size'],
        random_state=params['random_state'],
        n_trials=params['hyperparameter_trials']
    )

    auc_string = f"{auc:.4f}" if auc is not None else "N/A"
    log_info(f"Evaluation on holdout set (AUC={auc_string}):")
    log_info("\n" + report)

    output_path = Path(paths['model'])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    log_success(f"Saved model to {paths['model']}")
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    log_success(f"Saved metadata to {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced classification model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main_cli(args)