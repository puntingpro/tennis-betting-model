import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from scripts.utils.file_utils import load_dataframes
from scripts.utils.git_utils import get_git_hash
from scripts.utils.logger import log_info, log_success, setup_logging
from scripts.utils.schema import normalize_columns


def run_train_ev_filter_model(df: pd.DataFrame, min_ev: float):
    """
    Trains a simple logistic regression model to filter based on EV.
    """
    df = normalize_columns(df)
    
    # Create a binary target: 1 if the bet has a positive EV, 0 otherwise
    df["is_value_bet"] = (df["expected_value"] > min_ev).astype(int)
    
    features = ["predicted_prob", "odds"]
    X = df[features]
    y = df["is_value_bet"]
    
    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)
    
    report = classification_report(y, model.predict(X), output_dict=False)
    
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "model_type": "EVFilterLogisticRegression",
        "features": features,
        "min_ev_threshold": min_ev,
    }
    return model, report, meta


def main_cli(args):
    """
    Main CLI handler for training the EV filter model.
    """
    df = load_dataframes(args.input_glob)
    model, report, meta = run_train_ev_filter_model(df, min_ev=args.min_ev)
    
    log_info("EV Filter Model Training Report:")
    log_info("\n" + str(report))
    
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not getattr(args, 'dry_run', False):
        joblib.dump(model, output_path)
        log_success(f"Saved EV filter model to {args.output_model}")
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        log_success(f"Saved metadata to {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    setup_logging()
    # This part would need argparse setup to be runnable standalone
    # For now, it's designed to be called from main.py