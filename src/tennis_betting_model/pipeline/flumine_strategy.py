# src/tennis_betting_model/pipeline/flumine_strategy.py

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from betfairlightweight.resources.bettingresources import MarketBook
from flumine import BaseStrategy
from flumine.markets.market import Market
from flumine.order.order import OrderStatus
from flumine.order.ordertype import LimitOrder
from flumine.order.trade import Trade

# FIX: Imports moved to the top of the file to resolve Ruff E402 error
from ..pipeline.value_finder import MarketProcessor
from ..utils.common import get_tournament_category
from ..utils.logger import log_info, log_success, log_warning

# Set up a specific logger for this module
logger = logging.getLogger(__name__)


class TennisValueStrategy(BaseStrategy):
    """
    A Flumine strategy to identify and place value bets on tennis markets
    in near real-time using the Betfair Stream API.
    """

    def __init__(
        self,
        market_filter: dict,
        market_processor: MarketProcessor,
        betting_config: dict,
        live_trading_config: dict,
        dry_run: bool,
        processed_bets_log_path: str,
        **kwargs,
    ):
        super().__init__(market_filter=market_filter, **kwargs)
        self.market_processor = market_processor
        self.betting_config = betting_config
        self.live_trading_config = live_trading_config
        self.dry_run = dry_run

        # Staking parameters
        self.live_bankroll = float(self.betting_config.get("live_bankroll", 1000.0))
        self.fallback_bankroll = self.live_bankroll
        self.live_kelly_fraction = float(
            self.betting_config.get("live_kelly_fraction", 0.1)
        )
        self.max_kelly_stake_fraction = float(
            self.betting_config.get("max_kelly_stake_fraction", 0.1)
        )

        self.order_timeout_seconds = int(
            self.live_trading_config.get("order_timeout_seconds", 120)
        )

        # State Persistence
        self.processed_bets_log_path = Path(processed_bets_log_path)
        self.processed_selections = self._load_processed_selections()

        logger.info(
            f"TennisValueStrategy initialized. Dry Run: {self.dry_run}. "
            f"Loaded {len(self.processed_selections)} processed bets."
        )

    def _load_processed_selections(self) -> set:
        if self.processed_bets_log_path.exists():
            try:
                with open(self.processed_bets_log_path, "r") as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Error loading processed bets log: {e}. Starting with an empty set."
                )
                return set()
        return set()

    def _save_processed_selection(self, selection_key: str):
        self.processed_selections.add(selection_key)
        try:
            with open(self.processed_bets_log_path, "w") as f:
                json.dump(list(self.processed_selections), f)
        except IOError as e:
            logger.error(f"Error saving processed bets log: {e}")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if (
            not market.market_catalogue
            or not market.market_catalogue.competition
            or not market_book.market_definition
        ):
            return False

        if market_book.status != "OPEN" or market_book.inplay:
            return False

        profitable_tournaments = self.betting_config.get("profitable_tournaments", [])
        if profitable_tournaments:
            competition_name = getattr(market.market_catalogue.competition, "name", "")
            tournament_category = get_tournament_category(competition_name)
            if tournament_category not in profitable_tournaments:
                return False

        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = market_book.market_definition.market_time
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=datetime.timezone.utc)

        seconds_to_start = (start_time - now).total_seconds()

        # Only process markets starting within the next hour but not within the next minute
        if 60 < seconds_to_start < 3600:
            return True

        return False

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        value_bets = self.market_processor.process_market(
            market.market_catalogue, market_book
        )
        if value_bets:
            self.place_orders_from_bets(market, value_bets)

    def _get_live_bankroll(self) -> Optional[float]:
        client = self.clients.get_default()
        if client and client.account_funds:
            # FIX: Cast to float to resolve mypy no-any-return error
            return float(client.account_funds.available_to_bet_balance)
        return None

    def place_orders_from_bets(self, market: Market, value_bets: list):
        live_bankroll = self._get_live_bankroll()
        if live_bankroll is None:
            logger.warning(
                f"Live bankroll not available, using fallback value of ${self.fallback_bankroll:.2f}"
            )
            live_bankroll = self.fallback_bankroll

        if live_bankroll < 10:
            logger.warning(
                f"Live bankroll is very low (${live_bankroll:.2f}). Pausing placements."
            )
            return

        for bet in value_bets:
            selection_key = f"{market.market_id}-{bet['selection_id']}"
            if selection_key in self.processed_selections:
                continue
            if (
                market.blotter.selection_exposure(
                    self, (market.market_id, bet["selection_id"], 0)
                )
                != 0
            ):
                continue

            log_success(
                f"VALUE BET FOUND: {bet['player_name']} @ {bet['odds']} (EV: {bet['ev']})"
            )

            kelly_fraction = float(bet.get("kelly_fraction", 0.0))
            desired_kelly_fraction = kelly_fraction * self.live_kelly_fraction
            capped_kelly_fraction = min(
                desired_kelly_fraction, self.max_kelly_stake_fraction
            )
            stake = live_bankroll * capped_kelly_fraction

            if stake < 0.03:
                logger.warning(
                    f"Stake {stake:.2f} is below minimum for {bet['player_name']}, not placing bet."
                )
                self._save_processed_selection(selection_key)
                continue

            if self.dry_run:
                log_info(
                    f"[DRY RUN] Would place bet: {bet['player_name']} @ {bet['odds']} with stake ${stake:.2f} on {market.market_catalogue.event.name}"
                )
                self._save_processed_selection(selection_key)
                continue

            trade = Trade(
                market_id=market.market_id,
                selection_id=bet["selection_id"],
                handicap=0,
                strategy=self,
            )
            order = trade.create_order(
                side="BACK",
                order_type=LimitOrder(
                    price=bet["odds"], size=round(stake, 2), persistence_type="KEEP"
                ),
            )
            try:
                market.place_order(order)
                log_success(
                    f"PLACING STREAM BET: {bet['player_name']} @ {bet['odds']} with stake ${stake:.2f} on {market.market_catalogue.event.name}"
                )
                self._save_processed_selection(selection_key)
            except Exception as e:
                logger.error(
                    f"Failed to place order for {bet['player_name']}: {e}",
                    exc_info=True,
                )

    def process_orders(self, market: Market, orders: list):
        for order in orders:
            if (
                order.status == OrderStatus.EXECUTABLE
                and order.elapsed_seconds
                and order.elapsed_seconds > self.order_timeout_seconds
            ):
                log_warning(
                    f"Order {order.id} for selection {order.selection_id} has been executable for > {self.order_timeout_seconds}s. Cancelling."
                )
                try:
                    market.cancel_order(order)
                except Exception as e:
                    logger.error(
                        f"Failed to cancel order {order.id}: {e}", exc_info=True
                    )
