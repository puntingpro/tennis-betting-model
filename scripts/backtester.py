import pandas as pd
import joblib


def main():
    """
    Main function to run the backtesting process for SET betting markets using a
    robust merge strategy with the new set-specific features.
    """
    # --- 1. Load All Necessary Data ---
    print("\nLoading data sources for SET betting analysis...")
    try:
        odds_df = pd.read_csv("tennis_set_data.csv")
        model = joblib.load("models/advanced_xgb_set_model.joblib")
        # --- MODIFICATION: Use the new set-level feature file ---
        features_df = pd.read_csv("data/processed/all_advanced_set_features.csv")
        player_map = pd.read_csv("player_id_map.csv")
        print("All data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    # --- 2. Prepare Odds Data ---
    print("Preparing SET odds data...")
    odds_df["pt"] = pd.to_datetime(odds_df["pt"]).dt.tz_localize("UTC")
    odds_df["market_start_time"] = pd.to_datetime(odds_df["market_start_time"])

    pre_match_odds = odds_df[odds_df["pt"] < odds_df["market_start_time"]]
    last_odds_df = pre_match_odds.loc[
        pre_match_odds.groupby(["market_id", "runner_id"])["pt"].idxmax()
    ]

    market_groups = last_odds_df.groupby("market_id").filter(lambda x: len(x) == 2)
    pivoted_odds = market_groups.pivot_table(
        index="market_id",
        columns=market_groups.groupby("market_id").cumcount(),
        values=["runner_id", "best_back_price", "runner_status"],
        aggfunc={
            "runner_id": "first",
            "best_back_price": "mean",
            "runner_status": "first",
        },
    ).reset_index()
    pivoted_odds.columns = [
        f"{c[0]}_{c[1]}" if c[1] != "" else c[0] for c in pivoted_odds.columns
    ]

    # --- 3. Prepare Mapping and Feature Data ---
    print("Preparing features and mapping IDs...")
    id_map_dict = player_map.set_index("betfair_id")["feature_id"].to_dict()

    pivoted_odds["feature_p1_id"] = pivoted_odds["runner_id_0"].map(id_map_dict)
    pivoted_odds["feature_p2_id"] = pivoted_odds["runner_id_1"].map(id_map_dict)
    pivoted_odds.dropna(subset=["feature_p1_id", "feature_p2_id"], inplace=True)
    pivoted_odds["feature_p1_id"] = pivoted_odds["feature_p1_id"].astype(int)
    pivoted_odds["feature_p2_id"] = pivoted_odds["feature_p2_id"].astype(int)

    # --- 4. Merge Odds and Features ---
    print("Merging set odds data with set features...")
    # --- FINAL FIX: Merge directly on match_id ---
    final_df = pd.merge(
        pivoted_odds,
        features_df,
        # Match features to the correct player, regardless of p1/p2 assignment
        left_on=["feature_p1_id", "feature_p2_id"],
        right_on=["p1_id", "p2_id"],
        how="inner",
    )
    print(
        f"Successfully merged data, resulting in {len(final_df)} processable set markets."
    )

    if final_df.empty:
        print(
            "Analysis complete. No set markets could be matched between odds and feature files."
        )
        return

    # --- 5. Run Model Predictions and Analysis ---
    print("Running model predictions on set markets...")
    all_set_analysis = []

    for _, row in final_df.iterrows():
        model_features = model.feature_names_in_
        X = pd.DataFrame([row[model_features]], columns=model_features)

        p1_model_prob = model.predict_proba(X)[0][1]
        p2_model_prob = 1 - p1_model_prob

        p1_odds = row["best_back_price_0"]
        p2_odds = row["best_back_price_1"]
        p1_ev = (p1_model_prob * (p1_odds - 1)) - (1 - p1_model_prob)
        p2_ev = (p2_model_prob * (p2_odds - 1)) - (1 - p2_model_prob)

        winner_id = (
            row["runner_id_0"]
            if row["runner_status_0"] == "WINNER"
            else row["runner_id_1"]
        )

        all_set_analysis.append(
            {
                "market_id": row["market_id"],
                "p1_id": row["runner_id_0"],
                "p1_odds": p1_odds,
                "p1_model_prob": p1_model_prob,
                "p1_ev": p1_ev,
                "p2_id": row["runner_id_1"],
                "p2_odds": p2_odds,
                "p2_model_prob": p2_model_prob,
                "p2_ev": p2_ev,
                "winner_id": winner_id,
            }
        )

    # --- 6. Save Results ---
    analysis_df = pd.DataFrame(all_set_analysis)
    output_path = "set_betting_analysis.csv"
    analysis_df.to_csv(output_path, index=False)
    print(f"\nAnalysis complete. Processed {len(analysis_df)} set markets.")
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
