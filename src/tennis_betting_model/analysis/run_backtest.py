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
from tennis_betting_model.utils.betting_math import add_ev_and_kelly
from tennis_betting_model.utils.constants import BACKTEST_MAX_ODDS, BOOKMAKER_MARGIN


def run_simulation_backtest(df, model, ev_threshold, confidence_threshold):
    # This function remains unchanged
    log_info("Simulating odds based on Elo and finding value...")
    df["p1_predicted_prob"] = model.predict_proba(df[model.feature_names_in_])[:, 1]
    df["p2_predicted_prob"] = 1 - df["p1_predicted_prob"]
    df["p1_elo_prob"] = 1 / (1 + 10 ** ((df["p2_elo"] - df["p1_elo"]) / 400))
    df["p2_elo_prob"] = 1 - df["p1_elo_prob"]
    df["p1_odds"] = np.where(
        df["p1_elo_prob"] > 0,
        (1 / df["p1_elo_prob"]) / BOOKMAKER_MARGIN,
        BACKTEST_MAX_ODDS,
    )
    df["p1_odds"] = df["p1_odds"].clip(upper=BACKTEST_MAX_ODDS)
    df["p2_odds"] = np.where(
        df["p2_elo_prob"] > 0,
        (1 / df["p2_elo_prob"]) / BOOKMAKER_MARGIN,
        BACKTEST_MAX_ODDS,
    )
    df["p2_odds"] = df["p2_odds"].clip(upper=BACKTEST_MAX_ODDS)
    base_cols = ["match_id", "tourney_name", "tourney_date"]
    bets_p1 = df[base_cols + ["winner"]].copy()
    bets_p1["odds"] = df["p1_odds"]
    bets_p1["predicted_prob"] = df["p1_predicted_prob"]
    bets_p2 = df[base_cols + ["winner"]].copy()
    bets_p2["odds"] = df["p2_odds"]
    bets_p2["predicted_prob"] = df["p2_predicted_prob"]
    bets_p2["winner"] = 1 - bets_p2["winner"]
    bets_p1 = add_ev_and_kelly(bets_p1)
    bets_p2 = add_ev_and_kelly(bets_p2)
    value_bets_p1 = bets_p1[
        (bets_p1["expected_value"] > ev_threshold)
        & (bets_p1["predicted_prob"] > confidence_threshold)
    ]
    value_bets_p2 = bets_p2[
        (bets_p2["expected_value"] > ev_threshold)
        & (bets_p2["predicted_prob"] > confidence_threshold)
    ]
    return pd.concat([value_bets_p1, value_bets_p2], ignore_index=True)


def run_realistic_backtest(
    df, model, market_data_df, ev_threshold, confidence_threshold
):
    """
    REFACTOR: Runs backtest using pre-built historical market data.
    This function is now much simpler.
    """
    log_info("Merging model features with clean market data...")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"]).dt.date
    market_data_df["tourney_date"] = pd.to_datetime(
        market_data_df["tourney_date"]
    ).dt.date

    # Use historical_id from features (p1_id, p2_id) to merge with market data
    merged_df = pd.merge(df, market_data_df, on=["p1_id", "p2_id", "tourney_date"])

    if merged_df.empty:
        log_error("Could not merge any features with the backtest market data.")
        return pd.DataFrame()

    log_info(f"Successfully merged {len(merged_df)} markets. Making predictions...")
    merged_df["p1_predicted_prob"] = model.predict_proba(
        merged_df[model.feature_names_in_]
    )[:, 1]
    merged_df["p2_predicted_prob"] = 1 - merged_df["p1_predicted_prob"]

    # --- REFACTOR: Add 'tourney_name' to the output columns ---
    bets_p1 = (
        merged_df[["market_id", "tourney_name", "tourney_date", "winner_y"]]
        .copy()
        .rename(columns={"winner_y": "winner", "market_id": "match_id"})
    )
    bets_p1["odds"] = merged_df["p1_odds"]
    bets_p1["predicted_prob"] = merged_df["p1_predicted_prob"]

    bets_p2 = (
        merged_df[["market_id", "tourney_name", "tourney_date", "winner_y"]]
        .copy()
        .rename(columns={"winner_y": "winner", "market_id": "match_id"})
    )
    # --- END REFACTOR ---
    bets_p2["odds"] = merged_df["p2_odds"]
    bets_p2["predicted_prob"] = merged_df["p2_predicted_prob"]
    bets_p2["winner"] = 1 - bets_p2["winner"]

    bets_p1 = add_ev_and_kelly(bets_p1)
    bets_p2 = add_ev_and_kelly(bets_p2)

    value_bets_p1 = bets_p1[
        (bets_p1["expected_value"] > ev_threshold)
        & (bets_p1["predicted_prob"] > confidence_threshold)
    ]
    value_bets_p2 = bets_p2[
        (bets_p2["expected_value"] > ev_threshold)
        & (bets_p2["predicted_prob"] > confidence_threshold)
    ]

    return pd.concat([value_bets_p1, value_bets_p2], ignore_index=True)


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

    # Rename historical_id columns to p1_id and p2_id for merging
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

    if args.mode == "simulation":
        final_value_bets = run_simulation_backtest(
            features_df, model, ev_threshold, confidence_threshold
        )
    elif args.mode == "realistic":
        try:
            log_info(
                f"Loading clean backtest market data from {paths['backtest_market_data']}..."
            )
            market_data_df = pd.read_csv(
                paths["backtest_market_data"], parse_dates=["tourney_date"]
            )
            final_value_bets = run_realistic_backtest(
                features_df, model, market_data_df, ev_threshold, confidence_threshold
            )
        except FileNotFoundError:
            log_error(
                "Required file not found. Please ensure the 'build' command has been run to create backtest market data."
            )
            return
    else:
        log_error(f"Invalid mode '{args.mode}'. Choose 'simulation' or 'realistic'.")
        return

    if final_value_bets.empty:
        log_info("No value bets found for the selected mode.")
        return

    betfair_commission = betting_params.get("betfair_commission", 0.05)
    final_value_bets = final_value_bets.copy()
    final_value_bets["pnl"] = final_value_bets.apply(
        lambda row: (row["odds"] - 1) * (1 - betfair_commission)
        if row["winner"] == 1
        else -1,
        axis=1,
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
