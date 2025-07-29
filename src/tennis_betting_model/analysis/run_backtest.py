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


def run_simulation_backtest(df, model, ev_threshold):
    """Runs the backtest using Elo-simulated odds."""
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

    confidence_threshold = 0.55
    value_bets_p1 = bets_p1[
        (bets_p1["expected_value"] > ev_threshold)
        & (bets_p1["predicted_prob"] > confidence_threshold)
    ]
    value_bets_p2 = bets_p2[
        (bets_p2["expected_value"] > ev_threshold)
        & (bets_p2["predicted_prob"] > confidence_threshold)
    ]

    return pd.concat([value_bets_p1, value_bets_p2], ignore_index=True)


def get_winner_from_odds(group):
    """Determines the winner based on the runner with the lowest last traded price."""
    winner_row = group.loc[group["best_back_price"].idxmin()]
    return winner_row["feature_id"]


def run_realistic_backtest(df, model, odds_df, player_map_df, ev_threshold):
    """Runs the backtest using real historical Betfair odds."""
    log_info("Preparing real odds data for backtest...")

    last_prices = odds_df.loc[odds_df.groupby("market_id")["pt"].idxmax()]
    last_prices_mapped = pd.merge(
        last_prices, player_map_df, left_on="runner_id", right_on="betfair_id"
    )
    market_winners = (
        last_prices_mapped.groupby("market_id")
        .apply(get_winner_from_odds, include_groups=False)
        .reset_index(name="winner_id")
    )

    pre_match_odds = odds_df.loc[odds_df.groupby("market_id")["pt"].idxmin()]
    odds_with_feature_id = pd.merge(
        pre_match_odds, player_map_df, left_on="runner_id", right_on="betfair_id"
    )

    market_pairs = (
        odds_with_feature_id.groupby("market_id")["feature_id"]
        .apply(list)
        .reset_index()
    )
    market_pairs = market_pairs[market_pairs["feature_id"].apply(len) == 2]

    if market_pairs.empty:
        log_error("No markets found with two mapped players.")
        return pd.DataFrame()

    market_pairs[["p1_id_map", "p2_id_map"]] = pd.DataFrame(
        market_pairs["feature_id"].tolist(), index=market_pairs.index
    )
    market_pairs["p1_id"] = market_pairs[["p1_id_map", "p2_id_map"]].min(axis=1)
    market_pairs["p2_id"] = market_pairs[["p1_id_map", "p2_id_map"]].max(axis=1)

    p1_odds = pd.merge(
        market_pairs,
        odds_with_feature_id,
        left_on=["market_id", "p1_id"],
        right_on=["market_id", "feature_id"],
    )
    p1_odds = p1_odds.rename(
        columns={"best_back_price": "p1_odds", "pt": "tourney_date"}
    )

    p2_odds = pd.merge(
        market_pairs,
        odds_with_feature_id,
        left_on=["market_id", "p2_id"],
        right_on=["market_id", "feature_id"],
    )
    p2_odds = p2_odds.rename(columns={"best_back_price": "p2_odds"})

    market_odds_df = pd.merge(
        p1_odds[["market_id", "p1_id", "p2_id", "tourney_date", "p1_odds"]],
        p2_odds[["market_id", "p2_odds"]],
        on="market_id",
    )

    market_data = pd.merge(market_odds_df, market_winners, on="market_id")
    market_data["winner"] = (market_data["p1_id"] == market_data["winner_id"]).astype(
        int
    )

    log_info("Merging model features with unified Betfair market data...")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"]).dt.date
    market_data["tourney_date"] = pd.to_datetime(market_data["tourney_date"]).dt.date

    merged_df = pd.merge(df, market_data, on=["p1_id", "p2_id", "tourney_date"])

    if merged_df.empty:
        log_error("Could not merge any features with real odds data.")
        return pd.DataFrame()

    log_info(f"Successfully merged {len(merged_df)} markets. Making predictions...")
    merged_df["p1_predicted_prob"] = model.predict_proba(
        merged_df[model.feature_names_in_]
    )[:, 1]
    merged_df["p2_predicted_prob"] = 1 - merged_df["p1_predicted_prob"]

    bets_p1 = merged_df[["market_id", "tourney_date", "winner"]].copy()
    bets_p1["odds"] = merged_df["p1_odds"]
    bets_p1["predicted_prob"] = merged_df["p1_predicted_prob"]

    bets_p2 = merged_df[["market_id", "tourney_date", "winner"]].copy()
    bets_p2["odds"] = merged_df["p2_odds"]
    bets_p2["predicted_prob"] = merged_df["p2_predicted_prob"]
    bets_p2["winner"] = 1 - bets_p2["winner"]

    bets_p1 = add_ev_and_kelly(bets_p1)
    bets_p2 = add_ev_and_kelly(bets_p2)

    confidence_threshold = 0.55
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

    log_info(f"Loading model from {paths['model']}...")
    model = joblib.load(paths["model"])

    log_info(f"Loading historical features from {paths['consolidated_features']}...")
    features_df = pd.read_csv(
        paths["consolidated_features"], parse_dates=["tourney_date"]
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
            features_df, model, betting_params["ev_threshold"]
        )
    elif args.mode == "realistic":
        try:
            log_info(f"Loading real odds data from {paths['betfair_odds']}...")
            odds_df = pd.read_csv(paths["betfair_odds"], parse_dates=["pt"])

            log_info("Loading player ID map...")
            player_map_df = pd.read_csv(paths["player_map"])

            final_value_bets = run_realistic_backtest(
                features_df,
                model,
                odds_df,
                player_map_df,
                betting_params["ev_threshold"],
            )
        except FileNotFoundError:
            log_error(
                "Required file not found. Please ensure betfair_odds and player_map CSVs exist."
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
    final_value_bets.rename(columns={"market_id": "match_id"}, inplace=True)
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
