# FILE: src/tennis_betting_model/pipeline/value_finder.py

import pandas as pd
from decimal import Decimal
from typing import cast, Dict, Any

from ..utils.logger import log_warning
from ..builders.feature_builder import FeatureBuilder
from ..utils.config_schema import Betting


class MarketProcessor:
    """
    Encapsulates all logic for processing a single live betting market
    to find value bets. It relies on injected dependencies for feature building
    and model predictions.
    """

    def __init__(
        self, model: Any, feature_builder: FeatureBuilder, betting_config: Betting
    ):
        self.model = model
        self.feature_builder = feature_builder
        # Use Decimal for precise financial calculations from a typed config object
        self.ev_threshold = Decimal(str(betting_config.ev_threshold))

    def _check_player_for_value(
        self, market_catalogue, runner_meta, runner_book, win_prob
    ) -> dict | None:
        """Checks a single player/runner for a value bet."""
        if runner_book.ex.available_to_back:
            odds = Decimal(str(runner_book.ex.available_to_back[0]["price"]))
            ev = (win_prob * odds) - Decimal("1.0")
            if ev > self.ev_threshold:
                kelly_denominator = odds - Decimal("1.0")
                kelly = (
                    (ev / kelly_denominator)
                    if kelly_denominator > 0
                    else Decimal("0.0")
                )
                return self._create_bet_info(
                    market_catalogue,
                    runner_meta,
                    odds,
                    win_prob,
                    ev,
                    kelly,
                )
        return None

    def process_market(self, market_catalogue, market_book) -> list:
        """
        Analyzes a single market and returns any identified value bets.
        """
        if (
            not market_book
            or not hasattr(market_catalogue, "runners")
            or not hasattr(market_book, "runners")
            or len(market_catalogue.runners) != 2
            or len(market_book.runners) != 2
        ):
            return []

        try:
            p1_meta, p2_meta = market_catalogue.runners
            book_runners_dict = {r.selection_id: r for r in market_book.runners}
            p1_book = book_runners_dict.get(p1_meta.selection_id)
            p2_book = book_runners_dict.get(p2_meta.selection_id)

            if not p1_book or not p2_book:
                return []

            features = self._build_live_features(market_catalogue)
            if features is None:
                return []

            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(
                columns=self.model.feature_names_in_, fill_value=0
            )

            prediction = self.model.predict_proba(features_df)[0]
            win_prob_p1 = Decimal(str(prediction[1]))
            win_prob_p2 = Decimal("1.0") - win_prob_p1

            value_bets = []
            p1_bet = self._check_player_for_value(
                market_catalogue, p1_meta, p1_book, win_prob_p1
            )
            if p1_bet:
                value_bets.append(p1_bet)

            p2_bet = self._check_player_for_value(
                market_catalogue, p2_meta, p2_book, win_prob_p2
            )
            if p2_bet:
                value_bets.append(p2_bet)

            return value_bets
        except Exception as e:
            market_id = getattr(market_catalogue, "market_id", "UnknownID")
            log_warning(f"Skipping market {market_id} due to processing error: {e}")
            return []

    def _build_live_features(self, market_catalogue) -> dict | None:
        """Builds features for a live market using the injected feature_builder."""
        p1_meta, p2_meta = market_catalogue.runners
        try:
            p1_id, p2_id = int(p1_meta.selection_id), int(p2_meta.selection_id)
        except (ValueError, TypeError):
            return None

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
        return {
            "market_id": market.market_id,
            "selection_id": runner_meta.selection_id,
            "match": f"{comp_name} - {event_name}",
            "player_name": runner_meta.runner_name,
            "odds": float(odds),
            "model_prob": f"{prob:.2%}",
            "ev": f"{ev:+.2%}",
            "kelly_fraction": float(kelly) if kelly > 0 else 0.0,
        }
