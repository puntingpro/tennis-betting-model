# tests/pipeline/test_flumine_strategy.py
import unittest
import datetime
from unittest import mock
from pathlib import Path

from flumine.order.order import OrderStatus
from flumine.order.ordertype import LimitOrder
from flumine.order.trade import Trade

from src.tennis_betting_model.pipeline.flumine_strategy import TennisValueStrategy
from src.tennis_betting_model.utils.config_schema import Betting, LiveTradingParams
from src.tennis_betting_model.utils.constants import BetSide

# Import RiskManager to allow patching it


class FlumineStrategyTest(unittest.TestCase):
    """Unit tests for the TennisValueStrategy class."""

    @mock.patch("src.tennis_betting_model.pipeline.flumine_strategy.RiskManager")
    def setUp(self, mock_risk_manager):
        """Set up mock dependencies and strategy instance for each test."""
        self.mock_market_processor = mock.Mock()
        self.mock_betting_config = Betting(
            ev_threshold=0.1,
            confidence_threshold=0.52,
            betfair_commission=0.05,
            live_bankroll=1000.0,
            live_kelly_fraction=0.1,
            max_kelly_stake_fraction=0.05,
            profitable_tournaments=["Grand Slam"],
        )
        self.mock_live_trading_config = LiveTradingParams(
            poll_hours_ahead=12, order_timeout_seconds=120, stream_limit=195
        )
        self.mock_market_filter = {"filter": "mock"}

        self.db_path = Path("./test_processed_bets.db")
        if self.db_path.exists():
            self.db_path.unlink()

        # Instantiate the strategy. The RiskManager is now automatically mocked.
        self.strategy = TennisValueStrategy(
            market_filter=self.mock_market_filter,
            market_processor=self.mock_market_processor,
            betting_config=self.mock_betting_config,
            live_trading_config=self.mock_live_trading_config,
            dry_run=False,
            processed_bets_log_path=str(self.db_path),
        )

        # Keep a reference to the mocked RiskManager instance for use in tests
        self.mock_risk_manager_instance = mock_risk_manager.return_value
        self.strategy.risk_manager = self.mock_risk_manager_instance

        self.mock_client = mock.Mock()
        self.mock_client.account_funds.available_to_bet_balance = 1000.0
        self.strategy.clients = mock.Mock()
        self.strategy.clients.get_default.return_value = self.mock_client

    def tearDown(self):
        """Clean up the test database file after each test."""
        if self.db_path.exists():
            self.db_path.unlink()

    def _create_mock_market(self):
        """Helper to create a detailed mock market object for testing."""
        mock_market = mock.Mock()
        mock_market.market_catalogue.competition.name = "Wimbledon Grand Slam"
        mock_market.market_catalogue.event.name = "Player A v Player B"
        return mock_market

    def test_init(self):
        """Test strategy initialization."""
        self.assertTrue(self.db_path.exists())
        self.assertEqual(self.strategy.processed_selections, set())
        self.assertFalse(self.strategy.dry_run)

    def test_check_market_book_valid(self):
        """Test check_market_book returns True for a valid market."""
        mock_market = self._create_mock_market()
        mock_market_book = mock.Mock()
        mock_market_book.status = "OPEN"
        mock_market_book.inplay = False
        start_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=30
        )
        mock_market_book.market_definition.market_time = start_time
        self.assertTrue(self.strategy.check_market_book(mock_market, mock_market_book))

    def test_check_market_book_invalid_status(self):
        """Test check_market_book returns False for a closed market."""
        mock_market = self._create_mock_market()
        mock_market_book = mock.Mock(status="CLOSED")
        self.assertFalse(self.strategy.check_market_book(mock_market, mock_market_book))

    def test_check_market_book_inplay(self):
        """Test check_market_book returns False for an in-play market."""
        mock_market = self._create_mock_market()
        mock_market_book = mock.Mock(status="OPEN", inplay=True)
        self.assertFalse(self.strategy.check_market_book(mock_market, mock_market_book))

    def test_check_market_book_wrong_tournament(self):
        """Test check_market_book returns False for a non-profitable tournament."""
        mock_market = self._create_mock_market()
        mock_market.market_catalogue.competition.name = "Local Fun Event"
        mock_market_book = mock.Mock()
        mock_market_book.status = "OPEN"
        mock_market_book.inplay = False
        start_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=30
        )
        mock_market_book.market_definition.market_time = start_time
        self.assertFalse(self.strategy.check_market_book(mock_market, mock_market_book))

    def test_process_market_book(self):
        """Test that process_market_book calls dependencies correctly."""
        mock_market = mock.Mock()
        mock_market_book = mock.Mock()
        mock_value_bets = [{"some": "bet"}]
        self.mock_market_processor.process_market.return_value = mock_value_bets
        with mock.patch.object(
            self.strategy, "place_orders_from_bets"
        ) as mock_place_orders:
            self.strategy.process_market_book(mock_market, mock_market_book)
            self.mock_market_processor.process_market.assert_called_with(
                mock_market.market_catalogue, mock_market_book
            )
            mock_place_orders.assert_called_with(mock_market, mock_value_bets)

    def test_place_orders_from_bets(self):
        """Test the logic for placing an order from an identified value bet."""
        self.mock_risk_manager_instance.can_place_bet.return_value = True
        mock_market = mock.Mock()
        mock_market.market_id = "1.234"
        mock_market.blotter.selection_exposure.return_value = 0.0

        value_bets = [
            {
                "selection_id": 567,
                "player_name": "Test Player",
                "odds": 2.5,
                "ev": "20%",
                "kelly_fraction": 0.1,
            }
        ]

        with mock.patch.object(Trade, "create_order") as mock_create_order:
            self.strategy.place_orders_from_bets(mock_market, value_bets)

            stake = 10.0  # 1000 * 0.1 * 0.1

            mock_create_order.assert_called_once()
            call_args = mock_create_order.call_args[1]
            self.assertEqual(call_args["side"], BetSide.BACK.value)
            self.assertIsInstance(call_args["order_type"], LimitOrder)
            self.assertEqual(call_args["order_type"].price, 2.5)
            self.assertEqual(call_args["order_type"].size, round(stake, 2))

            mock_market.place_order.assert_called_once()
            self.assertIn("1.234-567", self.strategy.processed_selections)

    def test_place_orders_dry_run(self):
        """Test that no order is placed in dry-run mode."""
        self.strategy.dry_run = True
        self.mock_risk_manager_instance.can_place_bet.return_value = True
        mock_market = mock.Mock()
        mock_market.market_id = "1.234"
        mock_market.blotter.selection_exposure.return_value = 0.0
        value_bets = [
            {
                "selection_id": 567,
                "player_name": "Test Player",
                "odds": 2.5,
                "ev": "20%",
                "kelly_fraction": 0.1,
            }
        ]
        self.strategy.place_orders_from_bets(mock_market, value_bets)
        mock_market.place_order.assert_not_called()
        self.assertIn("1.234-567", self.strategy.processed_selections)

    def test_place_orders_already_processed(self):
        """Test that a bet on an already processed selection is not placed again."""
        mock_market = mock.Mock()
        mock_market.market_id = "1.234"
        selection_key = "1.234-567"
        self.strategy.processed_selections.add(selection_key)
        value_bets = [{"selection_id": 567}]
        self.strategy.place_orders_from_bets(mock_market, value_bets)
        mock_market.place_order.assert_not_called()

    def test_place_orders_risk_manager_blocks(self):
        """Test that the risk manager can block a bet."""
        self.mock_risk_manager_instance.can_place_bet.return_value = False
        mock_market = mock.Mock()
        value_bets = [
            {
                "selection_id": 567,
                "player_name": "Test Player",
                "odds": 2.5,
                "ev": "20%",
                "kelly_fraction": 0.1,
            }
        ]
        self.strategy.place_orders_from_bets(mock_market, value_bets)
        mock_market.place_order.assert_not_called()

    def test_process_orders_stale(self):
        """Test that stale, executable orders are cancelled."""
        mock_market = mock.Mock()
        mock_order = mock.Mock()
        mock_order.status = OrderStatus.EXECUTABLE
        mock_order.elapsed_seconds = self.strategy.order_timeout_seconds + 1

        self.strategy.process_orders(mock_market, [mock_order])
        mock_market.cancel_order.assert_called_with(mock_order)

    def test_process_orders_complete(self):
        """Test that settled orders update the risk manager's P&L."""
        mock_market = mock.Mock()
        mock_order = mock.Mock(status=OrderStatus.EXECUTION_COMPLETE, profit=10.0)

        self.strategy.process_orders(mock_market, [mock_order])
        self.mock_risk_manager_instance.update_pnl.assert_called_with(10.0)


if __name__ == "__main__":
    unittest.main()
