# NOTE: LightGBM was selected as the champion model after comparative backtesting
# against XGBoost. All modeling and training now defaults to LightGBM.

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path
import optuna
from typing import cast
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import log_info, log_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_lgbm(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    # FIX: Ignore mypy error for this line
    model = lgb.LGBMClassifier(**params, random_state=42)  # type: ignore
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    return cast(float, auc)


def train_eval_model(
    data: pd.DataFrame,
    model_output_path: str,
    n_trials: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    print("--- Starting Model Training for LightGBM ---")

    if data.empty:
        log_error("Feature DataFrame is empty. No data available to train the model.")
        log_error(
            "Please ensure your raw data contains settled matches and re-run the `build` command."
        )
        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(None, model_path)
        log_info(
            f"Empty placeholder model saved to {model_path} to allow pipeline completion."
        )
        return

    data.rename(
        columns=lambda c: c.replace("[", "").replace("]", "").replace("<", ""),
        inplace=True,
    )

    data["tourney_date"] = pd.to_datetime(data["tourney_date"])
    data = data.sort_values("tourney_date").reset_index(drop=True)

    print(
        f"Dataset sorted by date, ranging from {data['tourney_date'].min().date()} to {data['tourney_date'].max().date()}."
    )

    X = data.drop(
        columns=[
            "winner",
            "match_id",
            "p1_id",
            "p2_id",
            "tourney_date",
            "tourney_name",
            "surface",
        ],
        errors="ignore",
    )
    y = data["winner"]

    X = pd.get_dummies(X, columns=["p1_hand", "p2_hand"], drop_first=True)

    split_index = int(len(data) * (1 - test_size))
    X_train_main = X.iloc[:split_index]
    y_train_main = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    val_split_index = int(len(X_train_main) * (1 - 0.25))
    X_train = X_train_main.iloc[:val_split_index]
    y_train = y_train_main.iloc[:val_split_index]
    X_val = X_train_main.iloc[val_split_index:]
    y_val = y_train_main.iloc[val_split_index:]

    print(f"Training data: {len(X_train)} samples")
    print(f"Validation data: {len(X_val)} samples")
    print(f"Test data: {len(X_test)} samples")

    study = optuna.create_study(direction="maximize")
    print(f"Running Optuna optimization for {n_trials} trials...")

    study.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\nOptimization finished.")
    print(f"Best trial AUC: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(
        "\nTraining final model with best hyperparameters on the full training set..."
    )
    best_params = study.best_params
    # FIX: Ignore mypy error for this line
    final_model = lgb.LGBMClassifier(**best_params, random_state=42)  # type: ignore
    final_model.fit(X_train_main, y_train_main)

    print("Evaluating final model on the hold-out test set...")
    y_pred_final = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    report = classification_report(y_test, y_pred_final)

    print(f"\nFinal Model Test Accuracy: {final_accuracy:.4f}")
    print("Final Model Classification Report:")
    print(report)

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"\n✅ Final optimized LGBM model saved to {model_path}")


def main_cli(args):
    try:
        config = load_config(args.config)
        paths = config["data_paths"]
        model_params = config["model_params"]

        feature_path = Path(paths["consolidated_features"])

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature data not found at {feature_path}. Please run 'python main.py build' first."
            )

        log_info(f"Loading feature data from {feature_path}...")
        feature_data = pd.read_csv(feature_path, low_memory=False)

        max_samples = model_params.get("max_training_samples")
        if max_samples and len(feature_data) > max_samples:
            log_info(
                f"Full dataset has {len(feature_data)} rows. Taking a random sample of {max_samples} for faster optimization..."
            )
            feature_data = feature_data.sample(n=max_samples, random_state=42)

        train_eval_model(
            feature_data,
            model_output_path=paths["model"],
            n_trials=model_params["hyperparameter_trials"],
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
