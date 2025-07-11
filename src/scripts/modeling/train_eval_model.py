# src/scripts/modeling/train_eval_model.py

import sys
from pathlib import Path
import json
from datetime import datetime
import joblib
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import optuna
from optuna.integration import XGBoostPruningCallback

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.git_utils import get_git_hash
from src.scripts.utils.logger import log_info, log_success, setup_logging
from src.scripts.utils.config import load_config
from src.scripts.utils.schema import validate_data, PlayerFeaturesSchema

def train_advanced_model(df: pd.DataFrame, random_state: int, n_trials: int, n_splits: int = 5):
    df = validate_data(df, PlayerFeaturesSchema, "model_training_input")

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

    print(f"Training model with {len(features)} features using {n_splits}-fold cross-validation.")

    X = df_balanced[features]
    y = df_balanced['winner']

    def objective(trial):
        param = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': random_state
        }
        
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = XGBClassifier(**param, use_label_encoder=False)
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False, callbacks=[pruning_callback])
            
            y_prob = model.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y_test, y_prob))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    log_info(f"Best trial average AUC: {study.best_trial.value}")
    log_info(f"Best params: {study.best_params}")

    log_info("Training final model on the entire dataset with the best parameters...")
    final_model = XGBClassifier(**study.best_params, use_label_encoder=False, random_state=random_state)
    final_model.fit(X, y)

    # --- MODIFIED: Add feature importances to the metadata ---
    meta = {
        "timestamp": datetime.now().isoformat(), "git_hash": get_git_hash(),
        "model_type": type(final_model).__name__, "features": features,
        "feature_importances": final_model.feature_importances_.tolist(), # Convert to list for JSON
        "algorithm": "xgb", "train_rows": len(X), "cross_val_auc": study.best_trial.value,
        "best_params": study.best_params
    }
    return final_model, None, study.best_trial.value, meta

def main_cli(args):
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']
    params = config['model_params']

    print("Loading feature files...")
    df = pd.read_csv(paths['consolidated_features'])
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])

    model, report, auc, meta = train_advanced_model(
        df,
        random_state=params['random_state'],
        n_trials=params['hyperparameter_trials']
    )

    auc_string = f"{auc:.4f}" if auc is not None else "N/A"
    log_info(f"Final model trained. Best cross-validated AUC: {auc_string}")
    
    output_path = Path(paths['model'])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    log_success(f"Saved model to {paths['model']}")
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    log_success(f"Saved metadata to {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced classification model with cross-validation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main_cli(args)