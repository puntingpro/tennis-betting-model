import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from scripts.utils.file_utils import load_dataframes
from scripts.utils.git_utils import get_git_hash
from scripts.utils.logger import log_info, log_success, setup_logging
from scripts.utils.schema import normalize_columns, patch_winner_column


def run_train_eval_model(
    df: pd.DataFrame,
    algorithm: str = "rf",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple:
    """
    Train a classification model on value bets and evaluate.
    Returns (model, report, auc, meta_dict).
    """
    df = normalize_columns(df)
    df = patch_winner_column(df)
    
    if df.empty:
        raise ValueError("No valid data to train on after preprocessing.")

    excluded = {"winner", "match_id", "player_1", "player_2"}
    feature_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after preprocessing.")
    
    initial_rows = len(df)
    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise ValueError("All rows were dropped due to missing feature values.")
    log_info(f"Dropped {initial_rows - len(df)} rows with missing features.")
    
    log_info(f"Training model using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols]
    y = df["winner"]

    if y.nunique() < 2:
        raise ValueError(f"The target variable 'winner' has only {y.nunique()} unique value(s). Training cannot proceed.")

    if "match_id" not in df.columns:
        raise ValueError("'match_id' column is required for grouped train/test split.")
    groups = df["match_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if algorithm == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    elif algorithm == "logreg":
        model = LogisticRegression(max_iter=500, random_state=random_state, class_weight='balanced')
    elif algorithm == "xgb":
        # For XGBoost, you handle imbalance with 'scale_pos_weight'
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(eval_metric="logloss", random_state=random_state, scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, digits=3, output_dict=False)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "model_type": type(model).__name__,
        "features": feature_cols,
        "algorithm": algorithm,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "auc": auc,
    }
    return model, report, auc, meta


def main_cli(args):
    """
    Wrapper function to be called from the main CLI entrypoint.
    """
    df = load_dataframes(args.input_glob)
    model, report, auc, meta = run_train_eval_model(
        df, algorithm=args.algorithm, test_size=args.test_size
    )

    auc_string = f"{auc:.3f}" if auc is not None else "N/A"
    log_info(f"Evaluation on holdout set (AUC={auc_string}):")
    log_info("\n" + str(report))
    
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dry_run = getattr(args, 'dry_run', False)
    if not dry_run:
        joblib.dump(model, output_path)
        log_success(f"Saved model to {args.output_model}")
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        log_success(f"Saved metadata to {output_path.with_suffix('.json')}")