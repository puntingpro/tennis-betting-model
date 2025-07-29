import pandas as pd
from decimal import Decimal
from typing import cast, Dict, Any
from tennis_betting_model.utils.logger import log_info, log_success, log_warning
from tennis_betting_model.builders.feature_builder import FeatureBuilder
from tennis_betting_model.utils.alerter import alert_value_bets_found


class MarketProcessor:
    """
    Encapsulates all logic for processing a single live betting market
    to find value bets.
    """

    def __init__(self, model, feature_builder: FeatureBuilder, betting_config: dict):
        self.model = model
        self.feature_builder = feature_builder
        self.ev_threshold = Decimal(str(betting_config.get("ev_threshold", 0.1)))

    def process_market(self, market_catalogue, market_book) -> list:
        """
        Analyzes a single market and returns any identified value bets.
        """
        if (
            not market_book
            or len(market_catalogue.runners) != 2
            or len(market_book.runners) != 2
        ):
            return []

        try:
            p1_meta, p2_meta = market_catalogue.runners
            p1_book, p2_book = market_book.runners

            features = self._build_live_features(market_catalogue)
            if features is None:
                return []

            features_df = pd.DataFrame(
                [features], columns=self.model.feature_names_in_
            ).fillna(0)

            win_prob_p1 = Decimal(str(self.model.predict_proba(features_df)[0][1]))
            win_prob_p2 = Decimal("1.0") - win_prob_p1

            value_bets = []
            # Check player 1 for value
            if p1_book.ex.available_to_back:
                p1_odds = Decimal(str(p1_book.ex.available_to_back[0].price))
                p1_ev = (win_prob_p1 * p1_odds) - Decimal("1.0")
                if p1_ev > self.ev_threshold:
                    value_bets.append(
                        self._create_bet_info(
                            market_catalogue, p1_meta, p1_odds, win_prob_p1, p1_ev
                        )
                    )

            # Check player 2 for value
            if p2_book.ex.available_to_back:
                p2_odds = Decimal(str(p2_book.ex.available_to_back[0].price))
                p2_ev = (win_prob_p2 * p2_odds) - Decimal("1.0")
                if p2_ev > self.ev_threshold:
                    value_bets.append(
                        self._create_bet_info(
                            market_catalogue, p2_meta, p2_odds, win_prob_p2, p2_ev
                        )
                    )

            return value_bets

        except Exception as e:
            log_warning(
                f"Skipping market {market_catalogue.market_id} due to processing error: {e}"
            )
            return []

    def _build_live_features(self, market_catalogue) -> dict | None:
        """Builds features for a live market."""
        p1_meta, p2_meta = market_catalogue.runners

        try:
            p1_id, p2_id = int(p1_meta.selection_id), int(p2_meta.selection_id)
        except (ValueError, TypeError):
            log_warning(
                f"Invalid selection ID in market {market_catalogue.market_id}. Skipping."
            )
            return None

        surface = "Hard"
        if market_catalogue.market_name:
            name_lower = market_catalogue.market_name.lower()
            if "clay" in name_lower:
                surface = "Clay"
            elif "grass" in name_lower:
                surface = "Grass"

        match_date = pd.to_datetime(market_catalogue.market_start_time).tz_convert(
            "UTC"
        )

        features = self.feature_builder.build_features(
            p1_id, p2_id, surface, match_date
        )
        return cast(Dict[str, Any], features)

    def _create_bet_info(self, market, runner_meta, odds, prob, ev) -> dict:
        """Creates a formatted dictionary for an identified value bet."""
        bet_info = {
            "market_id": market.market_id,
            "selection_id": runner_meta.selection_id,
            "match": f"{market.competition.name} - {market.event.name}",
            "player_name": runner_meta.runner_name,
            "odds": float(odds),
            "model_prob": f"{prob:.2%}",
            "ev": f"{ev:+.2%}",
        }
        log_success(
            f"VALUE BET FOUND: {bet_info['player_name']} @ {bet_info['odds']} (EV: {bet_info['ev']})"
        )
        return bet_info


def process_markets(
    model,
    market_catalogues,
    market_book_lookup,
    player_info_lookup,
    df_rankings,
    df_matches,
    betting_config,
):
    """
    Builds features for live markets, makes predictions, and identifies value bets.
    """
    log_info(f"Processing {len(market_catalogues)} live markets...")

    feature_builder = FeatureBuilder(player_info_lookup, df_rankings, df_matches)
    market_processor = MarketProcessor(model, feature_builder, betting_config)

    all_value_bets = []
    for market in market_catalogues:
        market_book = market_book_lookup.get(market.market_id)
        value_bets_for_market = market_processor.process_market(market, market_book)
        if value_bets_for_market:
            all_value_bets.extend(value_bets_for_market)

    if all_value_bets:
        bet_df = pd.DataFrame(all_value_bets)
        alert_value_bets_found(bet_df)

    return all_value_bets
