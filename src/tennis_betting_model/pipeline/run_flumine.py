# FILE: src/tennis_betting_model/pipeline/run_flumine.py
import datetime
import joblib
import logging
from typing import List, Set

import betfairlightweight
import flumine
from betfairlightweight.resources import MarketCatalogue
from flumine.clients.betfairclient import BetfairClient
from flumine.streams.marketstream import MarketStream
from flumine.worker import BackgroundWorker

from ..builders.feature_builder import FeatureBuilder
from ..pipeline.flumine_strategy import TennisValueStrategy
from ..pipeline.value_finder import MarketProcessor
from ..utils.api import login_to_betfair
from ..utils.config import load_config, Config
from ..utils.data_loader import DataLoader
from ..utils.logger import log_error, log_info, log_warning, setup_logging

logger = logging.getLogger(__name__)


def fetch_and_limit_market_ids(
    client: betfairlightweight.APIClient, poll_filter: dict, limit: int
) -> Set[str]:
    logger.info("Fetching market catalogues via REST API...")
    try:
        api_client = client.betting if hasattr(client, "betting") else client
        market_catalogues: List[MarketCatalogue] = api_client.list_market_catalogue(
            filter=poll_filter,
            market_projection=["EVENT", "MARKET_START_TIME", "COMPETITION"],
            max_results=1000,
            sort="FIRST_TO_START",
        )
    except Exception as e:
        logger.error(f"Error during market catalogue fetching: {e}", exc_info=True)
        return set()
    if not market_catalogues:
        logger.info("No markets found matching the poll filter.")
        return set()
    valid_markets = [
        m
        for m in market_catalogues
        if hasattr(m, "market_start_time") and m.market_start_time
    ]
    if valid_markets:
        try:
            valid_markets.sort(key=lambda x: x.market_start_time)
        except TypeError:
            logger.warning("Could not sort markets due to unexpected start time types.")
    target_ids = set(m.market_id for m in valid_markets[:limit])
    logger.info(
        f"Identified {len(target_ids)} markets (out of {len(valid_markets)} found) for subscription."
    )
    return target_ids


def poll_markets(context: dict, flumine):
    logger.info("Worker: Starting market poll cycle...")
    lightweight_client = context.get("lightweight_client")
    poll_filter = context.get("poll_filter")
    strategy = context.get("strategy")
    stream_limit = context.get("stream_limit")

    if not all([lightweight_client, poll_filter, strategy, stream_limit]):
        logger.error("Worker: Missing required objects in context.")
        return

    # --- FIX: Assert types after the check to satisfy mypy ---
    assert poll_filter is not None
    assert strategy is not None
    assert stream_limit is not None

    target_market_ids = fetch_and_limit_market_ids(
        lightweight_client, poll_filter, stream_limit
    )
    old_stream = next((s for s in flumine.streams if isinstance(s, MarketStream)), None)
    if not old_stream:
        logger.warning("Worker: MarketStream not found. Will retry next cycle.")
        return
    current_market_ids = set(old_stream.market_filter.get("marketIds", []))
    if current_market_ids != target_market_ids:
        logger.info("Market subscription change detected. Restarting stream...")
        old_stream.stop()
        try:
            flumine.streams._streams.remove(old_stream)
            if old_stream in strategy.streams:
                strategy.streams.remove(old_stream)
        except ValueError:
            logger.warning("Worker: Old stream was already removed.")
        if not target_market_ids:
            logger.warning(
                "No target markets to subscribe to. Stream will remain stopped."
            )
            return
        new_market_filter = betfairlightweight.filters.market_filter(
            market_ids=list(target_market_ids)
        )
        new_stream = MarketStream(
            flumine=flumine,
            stream_id=None,
            market_filter=new_market_filter,
            market_data_filter=strategy.market_data_filter,
            streaming_timeout=strategy.streaming_timeout,
            conflate_ms=strategy.conflate_ms,
        )
        flumine.streams.add_custom_stream(new_stream)
        strategy.streams.append(new_stream)
        logger.info(
            f"Starting new MarketStream (ID: {new_stream.stream_id}) with {len(target_market_ids)} markets."
        )
        new_stream.start()
    else:
        logger.info(
            f"No subscription changes required. Tracking {len(current_market_ids)} markets."
        )


def main(args):
    setup_logging()
    logging.getLogger("flumine").setLevel(logging.INFO)
    logging.getLogger("betfairlightweight").setLevel(logging.WARNING)

    if args.dry_run:
        log_warning(
            "DRY-RUN MODE: Starting Real-Time Streaming Service. No bets will be placed."
        )
    else:
        log_info("LIVE MODE: Starting Real-Time Streaming Service.")

    try:
        config = Config(**load_config(args.config))
        log_info("Configuration file validation successful.")
    except Exception as e:
        log_error(f"Failed to load configuration: {e}. Exiting.")
        return

    try:
        lightweight_client = login_to_betfair(config.dict())
        log_info("Successfully logged in to Betfair.")
    except Exception as e:
        log_error(f"Failed to log in to Betfair: {e}. Exiting.")
        return

    poll_hours_ahead = config.live_trading_params.poll_hours_ahead
    stream_limit = config.live_trading_params.stream_limit
    now = datetime.datetime.utcnow()
    end_time = now + datetime.timedelta(hours=poll_hours_ahead)
    poll_filter = betfairlightweight.filters.market_filter(
        event_type_ids=["2"],
        market_type_codes=["MATCH_ODDS"],
        market_start_time={
            "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )

    log_info("Pre-polling markets to determine initial subscription list...")
    initial_market_ids = fetch_and_limit_market_ids(
        lightweight_client, poll_filter, stream_limit
    )

    log_info("Loading ML Model and supporting data...")
    try:
        model = joblib.load(config.data_paths.model)
        if model is None:
            log_error("Model file loaded but is empty (None). Cannot proceed.")
            return

        data_loader = DataLoader(config.data_paths)
        (
            df_matches,
            df_rankings,
            _,
            df_elo,
            player_info_lookup,
        ) = data_loader.load_all_pipeline_data()

        feature_builder = FeatureBuilder(
            player_info_lookup, df_rankings, df_matches, df_elo, config.elo_config
        )
        market_processor = MarketProcessor(model, feature_builder, config.betting)
    except Exception as e:
        log_error(
            f"Error loading pipeline data or initializing components: {e}. Exiting."
        )
        return

    client_wrapper = BetfairClient(lightweight_client)
    framework = flumine.Flumine(client=client_wrapper)
    initial_strategy_filter = betfairlightweight.filters.market_filter(
        market_ids=list(initial_market_ids)
    )
    streaming_data_filter = betfairlightweight.filters.streaming_market_data_filter(
        fields=["EX_BEST_OFFERS", "EX_MARKET_DEF"], ladder_levels=1
    )

    strategy = TennisValueStrategy(
        market_filter=initial_strategy_filter,
        market_processor=market_processor,
        betting_config=config.betting,
        live_trading_config=config.live_trading_params,
        dry_run=args.dry_run,
        processed_bets_log_path=config.data_paths.processed_bets_log,
        market_data_filter=streaming_data_filter,
    )
    framework.add_strategy(strategy)
    worker_context = {
        "lightweight_client": lightweight_client,
        "poll_filter": poll_filter,
        "strategy": strategy,
        "stream_limit": stream_limit,
    }
    framework.add_worker(
        BackgroundWorker(
            flumine=framework,
            function=poll_markets,
            interval=300,
            start_delay=5,
            name="poll_markets",
            context=worker_context,
        )
    )
    log_info(
        f"Starting Flumine framework. Worker active. Polling every 5 minutes for markets in the next {poll_hours_ahead} hours."
    )
    try:
        framework.run()
    except KeyboardInterrupt:
        log_info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        log_error(f"Flumine framework encountered a runtime error: {e}", exc_info=True)
