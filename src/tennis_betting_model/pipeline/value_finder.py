# FILE: src/tennis_betting_model/pipeline/value_finder.py

import pandas as pd
from decimal import Decimal
from typing import cast, Dict, Any

# Note: Using relative imports as standard for the project structure
from ..utils.logger import log_info, log_warning
from ..builders.feature_builder import FeatureBuilder
from ..utils.alerter import alert_value_bets_found


class MarketProcessor:
    """
    Encapsulates all logic for processing a single live betting market
    to find value bets.
    """

    def __init__(self, model, feature_builder: FeatureBuilder, betting_config: dict):
        self.model = model
        self.feature_builder = feature_builder
        # Use Decimal for precise financial calculations
        self.ev_threshold = Decimal(str(betting_config.get("ev_threshold", 0.1)))

    def process_market(self, market_catalogue, market_book) -> list:
        """
        Analyzes a single market and returns any identified value bets.
        """
        # Basic validation: ensure we have the book and exactly 2 runners
        if (
            not market_book
            or not hasattr(market_catalogue, "runners")
            or not hasattr(market_book, "runners")
            or len(market_catalogue.runners) != 2
            or len(market_book.runners) != 2
        ):
            return []

        try:
            # Extract metadata
            p1_meta, p2_meta = market_catalogue.runners

            # Ensure book runners align with catalogue runners (more robust than assuming order)
            book_runners_dict = {r.selection_id: r for r in market_book.runners}
            p1_book = book_runners_dict.get(p1_meta.selection_id)
            p2_book = book_runners_dict.get(p2_meta.selection_id)

            if not p1_book or not p2_book:
                # Handle case where runner data might be missing in the book update
                return []

            # Build features
            features = self._build_live_features(market_catalogue)
            if features is None:
                return []

            # Prepare features for the model
            features_df = pd.DataFrame([features])

            # Ensure the DataFrame columns exactly match the model's expected features, filling missing with 0.
            features_df = features_df.reindex(
                columns=self.model.feature_names_in_, fill_value=0
            )

            # Predict probabilities
            prediction = self.model.predict_proba(features_df)[0]
            win_prob_p1 = Decimal(str(prediction[1]))
            win_prob_p2 = Decimal("1.0") - win_prob_p1

            value_bets = []

            # Check Player 1 for value
            if p1_book.ex.available_to_back:
                p1_odds = Decimal(str(p1_book.ex.available_to_back[0]["price"]))
                p1_ev = (win_prob_p1 * p1_odds) - Decimal("1.0")
                if p1_ev > self.ev_threshold:
                    kelly_denominator = p1_odds - Decimal("1.0")
                    p1_kelly = (
                        (p1_ev / kelly_denominator)
                        if kelly_denominator > 0
                        else Decimal("0.0")
                    )
                    value_bets.append(
                        self._create_bet_info(
                            market_catalogue,
                            p1_meta,
                            p1_odds,
                            win_prob_p1,
                            p1_ev,
                            p1_kelly,
                        )
                    )

            # Check Player 2 for value
            if p2_book.ex.available_to_back:
                p2_odds = Decimal(str(p2_book.ex.available_to_back[0]["price"]))
                p2_ev = (win_prob_p2 * p2_odds) - Decimal("1.0")
                if p2_ev > self.ev_threshold:
                    kelly_denominator = p2_odds - Decimal("1.0")
                    p2_kelly = (
                        (p2_ev / kelly_denominator)
                        if kelly_denominator > 0
                        else Decimal("0.0")
                    )
                    value_bets.append(
                        self._create_bet_info(
                            market_catalogue,
                            p2_meta,
                            p2_odds,
                            win_prob_p2,
                            p2_ev,
                            p2_kelly,
                        )
                    )

            return value_bets

        except Exception as e:
            # Catch exceptions during processing of a single market to prevent crashing the whole stream
            market_id = getattr(market_catalogue, "market_id", "UnknownID")
            log_warning(
                f"Γ£ûΓîÅ Skipping market {market_id} due to processing error: {e}"
            )
            return []

    def _build_live_features(self, market_catalogue) -> dict | None:
        """Builds features for a live market."""
        p1_meta, p2_meta = market_catalogue.runners

        try:
            p1_id, p2_id = int(p1_meta.selection_id), int(p2_meta.selection_id)
        except (ValueError, TypeError):
            return None

        # Determine surface from market name
        surface = "Hard"
        if hasattr(market_catalogue, "market_name") and market_catalogue.market_name:
            name_lower = market_catalogue.market_name.lower()
            if "clay" in name_lower:
                surface = "Clay"
            elif "grass" in name_lower:
                surface = "Grass"

        match_date = pd.to_datetime(market_catalogue.market_start_time, utc=True)

        features = self.feature_builder.build_features(
            p1_id, p2_id, surface, match_date, match_id=market_catalogue.market_id
        )
        return cast(Dict[str, Any], features)

    def _create_bet_info(self, market, runner_meta, odds, prob, ev, kelly) -> dict:
        """Creates a formatted dictionary for an identified value bet."""

        comp_name = (
            getattr(market.competition, "name", "N/A")
            if hasattr(market, "competition")
            else "N/A"
        )
        event_name = (
            getattr(market.event, "name", "N/A") if hasattr(market, "event") else "N/A"
        )

        bet_info = {
            "market_id": market.market_id,
            "selection_id": runner_meta.selection_id,
            "match": f"{comp_name} - {event_name}",
            "player_name": runner_meta.runner_name,
            "odds": float(odds),
            "model_prob": f"{prob:.2%}",
            "ev": f"{ev:+.2%}",
            "kelly_fraction": float(kelly) if kelly > 0 else 0.0,
        }
        # Logging is now handled by the strategy to prevent duplicates
        return bet_info


def process_markets(
    model,
    market_catalogues,
    market_book_lookup,
    player_info_lookup,
    df_rankings,
    df_matches,
    df_elo,
    betting_config,
):
    """
    (LEGACY: Used by the polling pipeline 'run_pipeline.py')
    Builds features for live markets, makes predictions, and identifies value bets.
    """
    log_info(f"Processing {len(market_catalogues)} live markets...")

    feature_builder = FeatureBuilder(
        player_info_lookup, df_rankings, df_matches, df_elo
    )
    market_processor = MarketProcessor(model, feature_builder, betting_config)

    all_value_bets = []
    for market in market_catalogues:
        market_book = market_book_lookup.get(market.market_id)
        if market_book:
            value_bets_for_market = market_processor.process_market(market, market_book)
            if value_bets_for_market:
                all_value_bets.extend(value_bets_for_market)

    if all_value_bets:
        bet_df = pd.DataFrame(all_value_bets)
        alert_value_bets_found(bet_df)

    return all_value_bets
