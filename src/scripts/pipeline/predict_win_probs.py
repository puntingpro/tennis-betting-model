import pandas as pd

from scripts.utils.logger import log_info
from scripts.utils.schema import enforce_schema, normalize_columns


def predict_win_probs(model, df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    Adds win probability predictions to the input DataFrame.
    """
    df = normalize_columns(df)
    if features is None:
        features = getattr(
            model,
            "feature_names_in_",
            ["implied_prob_1", "implied_prob_2", "implied_prob_diff", "odds_margin"],
        )
    missing = [f for f in features if f not in df.columns]
    for f in missing:
        df[f] = pd.NA
    df_valid = df.dropna(subset=features)
    if df_valid.empty:
        empty_df = pd.DataFrame(
            columns=enforce_schema(pd.DataFrame(), "predictions").columns
        )
        return enforce_schema(empty_df, "predictions")
        
    if hasattr(model, "predict_proba"):
        df_valid["predicted_prob"] = model.predict_proba(df_valid[features])[:, 1]
    else:
        df_valid["predicted_prob"] = model.predict(df_valid[features])
    
    # Add the 'odds' column for the next stage
    if 'odds_1' in df_valid.columns:
        df_valid['odds'] = df_valid['odds_1']

    log_info("Added predicted_prob column.")
    return enforce_schema(df_valid, "predictions")


def main_cli():
    import argparse

    import joblib

    parser = argparse.ArgumentParser(description="Predict win probabilities")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    model = joblib.load(args.model_file)
    df = pd.read_csv(args.input_csv)
    result = predict_win_probs(model, df)
    if not args.dry_run:
        result.to_csv(args.output_csv, index=False)
        log_info(f"Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main_cli()