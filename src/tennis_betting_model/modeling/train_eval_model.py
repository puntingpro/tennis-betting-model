# src/tennis_betting_model/modeling/train_eval_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from pathlib import Path
from typing import cast
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import log_info, log_error, log_success

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_lgbm(
    trial: optuna.Trial, X_train, y_train, X_val, y_val, early_stopping_rounds: int
) -> float:
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
    model = lgb.LGBMClassifier(**params, random_state=42)  # type: ignore
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return cast(float, roc_auc_score(y_val, y_pred_proba))


def train_eval_model(
    data: pd.DataFrame,
    model_output_path: str,
    plot_dir: str,
    training_params: dict,
    test_size: float = 0.2,
):
    log_info("--- Starting Model Training for LightGBM ---")
    max_samples = training_params.get("max_training_samples")
    if max_samples and len(data) > max_samples:
        log_info(f"Taking a random sample of {max_samples} for faster optimization...")
        data = data.sample(n=max_samples, random_state=42)
    if data.empty:
        log_error("Feature DataFrame is empty. Cannot train model.")
        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(None, model_path)
        log_info(f"Empty placeholder model saved to {model_path}.")
        return
    data.rename(
        columns=lambda c: c.replace("[", "").replace("]", "").replace("<", ""),
        inplace=True,
    )
    data["tourney_date"] = pd.to_datetime(data["tourney_date"])
    data = data.sort_values("tourney_date").reset_index(drop=True)

    # --- FIX START: Add 'market_id' to the list of columns to drop ---
    X = data.drop(
        columns=[
            "winner",
            "market_id",  # Previously was "match_id", now explicitly dropping the correct column name
            "p1_id",
            "p2_id",
            "tourney_date",
            "tourney_name",
            "surface",
        ],
        errors="ignore",
    )
    # --- FIX END ---

    y = data["winner"]
    X = pd.get_dummies(X, columns=["p1_hand", "p2_hand"], drop_first=True)
    split_index = int(len(data) * (1 - test_size))
    X_train_main, y_train_main = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    validation_size = training_params.get("validation_size", 0.25)
    val_split_index = int(len(X_train_main) * (1 - validation_size))
    X_train, y_train = (
        X_train_main.iloc[:val_split_index],
        y_train_main.iloc[:val_split_index],
    )
    X_val, y_val = (
        X_train_main.iloc[val_split_index:],
        y_train_main.iloc[val_split_index:],
    )

    early_stopping_rounds = training_params.get("early_stopping_rounds", 50)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_lgbm(
            trial, X_train, y_train, X_val, y_val, early_stopping_rounds
        ),
        n_trials=training_params["hyperparameter_trials"],
        show_progress_bar=True,
    )
    log_info(f"Best trial AUC: {study.best_value:.4f}")
    final_model = lgb.LGBMClassifier(**study.best_params, random_state=42)
    final_model.fit(X_train_main, y_train_main)
    y_pred_final = final_model.predict(X_test)
    log_info(f"\nFinal Model Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    log_info(
        "Final Model Classification Report:\n"
        + classification_report(y_test, y_pred_final)
    )
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    log_success(f"Final optimized LGBM model saved to {model_path}")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        lgb.plot_importance(final_model, ax=ax, max_num_features=20)
        plt.tight_layout()
        plot_path = Path(plot_dir) / "feature_importance.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300)
        log_success(f"Feature importance plot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        log_error(f"Could not generate feature importance plot: {e}")


def main_cli(args):
    try:
        config = load_config(args.config)
        paths, training_params = config["data_paths"], config["training_params"]
        feature_path = Path(paths["consolidated_features"])
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature data not found at {feature_path}.")
        log_info(f"Loading feature data from {feature_path}...")
        feature_data = pd.read_csv(feature_path, low_memory=False)
        train_eval_model(
            feature_data,
            model_output_path=paths["model"],
            plot_dir=paths["plot_dir"],
            training_params=training_params,
        )
    except Exception as e:
        log_error(f"An error occurred during model training: {e}", exc_info=True)
