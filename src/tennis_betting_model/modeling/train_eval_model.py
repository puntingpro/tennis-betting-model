import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path
import optuna
from typing import cast
from tennis_betting_model.utils.config import load_config
import argparse


optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }

    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
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
    print("--- Starting Model Training and Hyperparameter Optimization ---")

    data.rename(
        columns=lambda c: c.replace("[", "").replace("]", "").replace("<", ""),
        inplace=True,
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

    X_train_main, X_test, y_train_main, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_main,
        y_train_main,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_main,
    )

    study = optuna.create_study(direction="maximize")
    print(f"Running Optuna optimization for {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
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
    final_model = xgb.XGBClassifier(
        **best_params, use_label_encoder=False, eval_metric="logloss", random_state=42
    )
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
    print(f"\n✅ Final optimized model saved to {model_path}")


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

        print(f"Loading feature data from {feature_path}...")
        feature_data = pd.read_csv(feature_path, low_memory=False)

        # Use max_training_samples from config if it exists
        max_samples = model_params.get("max_training_samples")
        if max_samples and len(feature_data) > max_samples:
            print(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the XGBoost model."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    cli_args = parser.parse_args()
    main_cli(cli_args)
