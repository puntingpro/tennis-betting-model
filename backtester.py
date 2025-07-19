import pandas as pd
import joblib


def load_and_prepare_data(csv_path="tennis_data.csv"):
    """Loads and prepares the historical odds data."""
    print(f"Loading historical odds from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["pt"] = pd.to_datetime(df["pt"]).dt.tz_localize("UTC")
    df["market_start_time"] = pd.to_datetime(df["market_start_time"])
    print("Odds data loaded successfully.")
    return df


def get_pre_match_odds(market_df):
    """Finds the last recorded odds for each runner before the market start time."""
    pre_match_df = market_df[market_df["pt"] < market_df["market_start_time"]]
    if pre_match_df.empty:
        return None
    last_odds = pre_match_df.groupby("runner_id").last()
    return last_odds if len(last_odds) == 2 else None


def get_match_winner(market_df):
    """Determines the winner of the match based on the final runner status."""
    final_status = market_df.groupby("runner_id").last()
    winner = final_status[final_status["runner_status"] == "WINNER"]
    return winner.index[0] if not winner.empty else None


def main():
    """Main function to run the final backtesting process with real features."""
    odds_df = load_and_prepare_data()
    if odds_df is None:
        return

    # --- Load Model and All Historical Features ---
    print("\nLoading XGBoost model and historical features...")
    try:
        model = joblib.load("models/advanced_xgb_model.joblib")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file 'models/advanced_xgb_model.joblib' not found.")
        return

    try:
        features_df = pd.read_csv("data/processed/all_advanced_features.csv")
        features_df.set_index("match_id", inplace=True)
        print("Historical features loaded successfully.")
    except FileNotFoundError:
        print(
            "Error: Features file 'data/processed/all_advanced_features.csv' not found."
        )
        return

    markets = odds_df.groupby("market_id")
    print(f"\nFound {len(markets)} unique markets to analyze. Running backtest...")

    betting_results = []
    EV_THRESHOLD = 0

    for market_id, market_df in markets:
        pre_match_odds_df = get_pre_match_odds(market_df)

        if pre_match_odds_df is not None:
            p1_id = int(pre_match_odds_df.index[0])
            p1_odds = pre_match_odds_df["best_back_price"].iloc[0]

            p2_id = int(pre_match_odds_df.index[1])
            p2_odds = pre_match_odds_df["best_back_price"].iloc[1]

            # --- Real Feature Lookup using match_id and player_ids ---
            # Find the match in the features dataframe where the players match, in any order
            try:
                match_features_row = features_df[
                    ((features_df["p1_id"] == p1_id) & (features_df["p2_id"] == p2_id))
                    | (
                        (features_df["p1_id"] == p2_id)
                        & (features_df["p2_id"] == p1_id)
                    )
                ]
                if match_features_row.empty:
                    continue  # Skip if we don't have features

                match_features = match_features_row.iloc[0]
                p1_is_first_in_features = match_features["p1_id"] == p1_id

            except (KeyError, IndexError):
                continue

            model_features = model.feature_names_in_
            X = pd.DataFrame([match_features[model_features]], columns=model_features)

            # --- Model Prediction ---
            p1_model_prob = model.predict_proba(X)[0][1]

            p1_win_prob = (
                p1_model_prob if p1_is_first_in_features else (1 - p1_model_prob)
            )
            p2_win_prob = 1 - p1_win_prob

            # --- Value Bet Calculation ---
            p1_ev = (p1_win_prob * (p1_odds - 1)) - (1 - p1_win_prob)
            p2_ev = (p2_win_prob * (p2_odds - 1)) - (1 - p2_win_prob)

            bet_on, bet_odds = (None, 0)
            if p1_ev > EV_THRESHOLD:
                bet_on, bet_odds = p1_id, p1_odds
            elif p2_ev > EV_THRESHOLD:
                bet_on, bet_odds = p2_id, p2_odds

            if bet_on:
                winner_id = get_match_winner(market_df)
                if winner_id is not None:
                    pnl = (bet_odds - 1) if bet_on == winner_id else -1
                    betting_results.append({"market_id": market_id, "pnl": pnl})

    if not betting_results:
        print("\nBacktest complete. No value bets were identified.")
        return

    results_df = pd.DataFrame(betting_results)
    total_bets = len(results_df)
    total_pnl = results_df["pnl"].sum()
    roi = (total_pnl / total_bets) * 100

    print("\n--- Final Backtesting Report ---")
    print(f"Total Bets Placed: {total_bets}")
    print(f"Total Profit/Loss: {total_pnl:.2f} units")
    print(f"Return on Investment (ROI): {roi:.2f}%")


if __name__ == "__main__":
    main()
