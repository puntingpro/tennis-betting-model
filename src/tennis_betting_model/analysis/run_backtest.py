import pandas as pd
import joblib
import argparse
import numpy as np
from pathlib import Path
from tennis_betting_model.utils.logger import (
    log_info,
    log_success,
    log_error,
    setup_logging,
)
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.betting_math import add_ev_and_kelly, calculate_pnl
from tennis_betting_model.utils.constants import BACKTEST_MAX_ODDS, BOOKMAKER_MARGIN


def run_backtest(
    df, model, ev_threshold, confidence_threshold, mode, market_data_df=None
):
    """
    A unified function to run backtests in either 'simulation' or 'realistic' mode.
    """
    if mode == "simulation":
        log_info("Simulating odds based on Elo and finding value...")
        df["p1_elo_prob"] = 1 / (1 + 10 ** ((df["p2_elo"] - df["p1_elo"]) / 400))
        df["p2_elo_prob"] = 1 - df["p1_elo_prob"]
        df["p1_odds"] = np.where(
            df["p1_elo_prob"] > 0,
            (1 / df["p1_elo_prob"]) / BOOKMAKER_MARGIN,
            BACKTEST_MAX_ODDS,
        ).clip(upper=BACKTEST_MAX_ODDS)
        df["p2_odds"] = np.where(
            df["p2_elo_prob"] > 0,
            (1 / df["p2_elo_prob"]) / BOOKMAKER_MARGIN,
            BACKTEST_MAX_ODDS,
        ).clip(upper=BACKTEST_MAX_ODDS)

        bets_df = df.copy()
        bets_df.rename(columns={"match_id": "market_id"}, inplace=True)

    elif mode == "realistic":
        log_info("Merging model features with clean market data...")
        df["tourney_date"] = pd.to_datetime(df["tourney_date"]).dt.date
        market_data_df["tourney_date"] = pd.to_datetime(
            market_data_df["tourney_date"]
        ).dt.date

        bets_df = pd.merge(df, market_data_df, on=["p1_id", "p2_id", "tourney_date"])
        if bets_df.empty:
            log_error("Could not merge any features with the backtest market data.")
            return pd.DataFrame()
        log_info(f"Successfully merged {len(bets_df)} markets. Making predictions...")

        # Use winner data from the market file
        bets_df.rename(columns={"winner_y": "winner"}, inplace=True)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'simulation' or 'realistic'.")

    # --- Common Logic: Predictions and Value Calculation ---
    bets_df["p1_predicted_prob"] = model.predict_proba(
        bets_df[model.feature_names_in_]
    )[:, 1]
    bets_df["p2_predicted_prob"] = 1 - bets_df["p1_predicted_prob"]

    base_cols = ["market_id", "tourney_name", "tourney_date", "winner"]

    bets_p1 = bets_df[base_cols].copy()
    bets_p1["odds"] = bets_df["p1_odds"]
    bets_p1["predicted_prob"] = bets_df["p1_predicted_prob"]

    bets_p2 = bets_df[base_cols].copy()
    bets_p2["odds"] = bets_df["p2_odds"]
    bets_p2["predicted_prob"] = bets_df["p2_predicted_prob"]
    bets_p2["winner"] = 1 - bets_p2["winner"]

    all_bets = pd.concat(
        [add_ev_and_kelly(bets_p1), add_ev_and_kelly(bets_p2)], ignore_index=True
    )

    value_bets = all_bets[
        (all_bets["expected_value"] > ev_threshold)
        & (all_bets["predicted_prob"] > confidence_threshold)
    ]
    return value_bets


def main(args):
    setup_logging()
    config = load_config(args.config)
    paths = config["data_paths"]
    betting_params = config["betting"]

    ev_threshold = betting_params["ev_threshold"]
    confidence_threshold = betting_params["confidence_threshold"]

    log_info(f"Loading model from {paths['model']}...")
    model = joblib.load(paths["model"])

    log_info(f"Loading historical features from {paths['consolidated_features']}...")
    features_df = pd.read_csv(
        paths["consolidated_features"], parse_dates=["tourney_date"]
    )
    features_df.rename(
        columns={"winner_historical_id": "p1_id", "loser_historical_id": "p2_id"},
        inplace=True,
    )

    features_df.rename(
        columns=lambda c: c.replace("[", "").replace("]", "").replace("<", ""),
        inplace=True,
    )
    features_df_dummies = pd.get_dummies(
        features_df, columns=["p1_hand", "p2_hand"], drop_first=True
    )
    missing_cols = set(model.feature_names_in_) - set(features_df_dummies.columns)
    for c in missing_cols:
        features_df_dummies[c] = 0
    features_df[model.feature_names_in_] = features_df_dummies[
        model.feature_names_in_
    ].fillna(0)

    market_data_df = None
    if args.mode == "realistic":
        try:
            log_info(
                f"Loading clean backtest market data from {paths['backtest_market_data']}..."
            )
            market_data_df = pd.read_csv(
                paths["backtest_market_data"], parse_dates=["tourney_date"]
            )
        except FileNotFoundError:
            log_error(
                "Required file not found. Please run the 'build' command to create backtest market data."
            )
            return

    final_value_bets = run_backtest(
        features_df,
        model,
        ev_threshold,
        confidence_threshold,
        args.mode,
        market_data_df,
    )

    if final_value_bets.empty:
        log_info("No value bets found for the selected mode.")
        return

    betfair_commission = betting_params.get("betfair_commission", 0.05)
    final_value_bets = calculate_pnl(
        final_value_bets.copy(), commission=betfair_commission
    )

    total_bets = len(final_value_bets)
    total_pnl = final_value_bets["pnl"].sum()
    roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0

    log_success(f"Found {total_bets} total historical value bets.")
    log_success(f"Total Profit/Loss: {total_pnl:.2f} units")
    log_success(f"Return on Investment (ROI): {roi:.2f}%")

    output_path = Path(paths["backtest_results"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_value_bets.to_csv(output_path, index=False)
    log_success(f"Saved final backtest results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a unified backtest for the tennis model."
    )
    parser.add_argument(
        "mode", choices=["simulation", "realistic"], help="The backtesting mode to run."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    cli_args = parser.parse_args()
    main(cli_args)
