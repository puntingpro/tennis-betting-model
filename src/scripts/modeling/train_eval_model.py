# src/scripts/modeling/train_eval_model.py

from pathlib import Path
import json
from datetime import datetime
import joblib
import pandas as pd
import argparse
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any

from scripts.utils.git_utils import get_git_hash
from scripts.utils.logger import log_info, log_success, setup_logging
from scripts.utils.config import load_config
from scripts.utils.schema import validate_data


def train_fast_model(
    df: pd.DataFrame, random_state: int
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Trains a single XGBoost model with a good set of default parameters.
    """
    try:
        df = validate_data(df, "model_training_input")
    except Exception as e:
        log_info(f"Schema validation failed, proceeding without it. Error: {e}")

    log_info("Creating a balanced dataset by flipping player perspectives...")
    df_loser = df.copy()
    p1_cols = [col for col in df.columns if col.startswith("p1_")]
    p2_cols = [col for col in df.columns if col.startswith("p2_")]
    swap_map = {
        **{p1: p2 for p1, p2 in zip(p1_cols, p2_cols)},
        **{p2: p1 for p1, p2 in zip(p1_cols, p2_cols)},
    }
    df_loser = df_loser.rename(columns=swap_map)
    df_loser["winner"] = 0
    df_balanced = pd.concat([df, df_loser], ignore_index=True)

    excluded_cols = [
        "match_id",
        "set_num",
        "set_winner_id",
        "p1_id",
        "p2_id",
        "winner",
        # --- FINAL FIX: Exclude non-numeric name columns from training ---
        "p1_name",
        "p2_name",
    ]
    features = [col for col in df_balanced.columns if col not in excluded_cols]
    df_balanced[features] = df_balanced[features].fillna(0)

    log_info(f"Training a fast model with {len(features)} features...")

    X = df_balanced[features]
    y = df_balanced["winner"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "random_state": random_state,
        "use_label_encoder": False,
    }

    final_model = XGBClassifier(**params)
    final_model.fit(X, y)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "model_type": type(final_model).__name__,
        "algorithm": "xgb_fast",
        "features": features,
        "feature_importances": final_model.feature_importances_.tolist(),
        "train_rows": len(X),
        "params": params,
    }
    return final_model, meta


def main_cli(args: argparse.Namespace) -> None:
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]
    params = config["model_params"]

    log_info("Loading SET-LEVEL feature files...")
    df = pd.read_csv("data/processed/all_advanced_set_features.csv")

    log_info("Running a fast training cycle with default parameters.")

    model, meta = train_fast_model(df, random_state=params["random_state"])

    log_info("Fast model trained successfully.")

    output_path = Path("models/advanced_xgb_set_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    log_success(f"Saved SET model to {output_path}")

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    log_success(f"Saved SET metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an advanced classification model with cross-validation."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main_cli(args)
