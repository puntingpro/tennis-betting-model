# src/tennis_betting_model/modeling/train_eval_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import cast
import json
from src.tennis_betting_model.utils.config_schema import Config
from src.tennis_betting_model.utils.logger import log_info, log_error, log_success

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_lgbm(
    trial: optuna.Trial, X_train, y_train, X_val, y_val, early_stopping_rounds: int
) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    if params["boosting_type"] == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

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
    perform_cv: bool = False,
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

    # Exclude hand columns from being dropped, so they can be one-hot encoded
    hand_cols = ["p1_hand", "p2_hand"]
    non_feature_cols = [
        col
        for col in data.columns
        if data[col].dtype == "object" and col not in hand_cols
    ]

    # FIX: Add the newly identified leaky columns from the feature importance plot
    identifier_cols = [
        "winner",
        "market_id",
        "p1_id",
        "p2_id",
        "tourney_date",
        "winner_id",
        "winner_historical_id",
        "loser_id",
        "loser_historical_id",
        "p1_tourney_date",
        "p2_tourney_date",
        "p1_won",
        "p2_won",
        "p1_winner_id",
        "p1_loser_id",
        "p2_winner_id",
        "p2_loser_id",
    ]
    cols_to_drop = list(set(non_feature_cols + identifier_cols))

    X = data.drop(columns=cols_to_drop, errors="ignore")
    y = data["winner"]

    # Now this step will work correctly as the hand columns are present in X
    X = pd.get_dummies(X, columns=hand_cols, drop_first=True)

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
    y_pred_proba_final = final_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred_final)
    roc_auc = roc_auc_score(y_test, y_pred_proba_final)
    report = classification_report(y_test, y_pred_final, output_dict=True)

    log_info(f"\nFinal Model Test Accuracy: {accuracy:.4f}")
    log_info(f"Final Model Test AUC: {roc_auc:.4f}")
    log_info(
        "Final Model Classification Report:\n"
        + classification_report(y_test, y_pred_final)
    )

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    log_success(f"Final optimized LGBM model saved to {model_path}")

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "best_params": study.best_params,
    }
    metrics_path = model_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log_success(f"Model performance metrics saved to {metrics_path}")

    if perform_cv:
        log_info("\n--- Performing Cross-Validation ---")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_main, y_train_main):
            X_train_cv, X_val_cv = (
                X_train_main.iloc[train_idx],
                X_train_main.iloc[val_idx],
            )
            y_train_cv, y_val_cv = (
                y_train_main.iloc[train_idx],
                y_train_main.iloc[val_idx],
            )

            model_cv = lgb.LGBMClassifier(**study.best_params, random_state=42)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_proba_cv = model_cv.predict_proba(X_val_cv)[:, 1]
            cv_scores.append(roc_auc_score(y_val_cv, y_pred_proba_cv))

        log_info(f"Cross-validation AUC scores: {[f'{s:.4f}' for s in cv_scores]}")
        log_info(
            f"Mean CV AUC: {pd.Series(cv_scores).mean():.4f} (+/- {pd.Series(cv_scores).std():.4f})"
        )

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


def main_cli(config: Config):
    try:
        paths, training_params = config.data_paths, config.training_params
        feature_path = Path(paths.consolidated_features)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature data not found at {feature_path}.")
        log_info(f"Loading feature data from {feature_path}...")
        feature_data = pd.read_csv(feature_path, low_memory=False)
        train_eval_model(
            feature_data,
            model_output_path=paths.model,
            plot_dir=paths.plot_dir,
            training_params=training_params.dict(),
        )
    except Exception as e:
        log_error(f"An error occurred during model training: {e}", exc_info=True)
