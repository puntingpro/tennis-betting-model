# src/scripts/utils/api.py
import os
import time
import requests
import betfairlightweight
from betfairlightweight.exceptions import APIError
from betfairlightweight.resources import MarketBook, MarketCatalogue
from typing import List, Tuple, Dict
# --- FIXED: Added log_error to the import statement ---
from .logger import log_info, log_warning, log_error

def login_to_betfair(config: dict) -> betfairlightweight.APIClient:
    """
    Logs in to the Betfair API, retrying on failure with exponential backoff.

    Args:
        config (dict): A dictionary containing configuration details.

    Returns:
        betfairlightweight.APIClient: An authenticated API client instance.
    """
    session = requests.Session()
    proxy_url = os.getenv('PROXY_URL')

    if proxy_url:
        log_info(f"Connecting via proxy...")
        proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }
        session.proxies.update(proxies)

    trading = betfairlightweight.APIClient(
        username=os.getenv('BF_USER'),
        password=os.getenv('BF_PASS'),
        app_key=os.getenv('BF_APP_KEY'),
        certs='certs/',
        session=session
    )

    # --- ADDED: Retry logic for login ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            trading.login()
            log_info("✅ Successfully logged in to Betfair.")
            return trading
        except APIError as e:
            log_warning(f"Login attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                log_info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                log_error("❌ All login attempts failed. Aborting.")
                raise  # Re-raise the final exception

def get_tennis_competitions(trading: betfairlightweight.APIClient, target_keywords: List[str]) -> List[str]:
    """
    Fetches and filters tennis competitions by a list of keywords.

    Args:
        trading (betfairlightweight.APIClient): The authenticated API client.
        target_keywords (List[str]): A list of keywords to filter competition names.

    Returns:
        List[str]: A list of competition IDs that match the keywords.
    """
    tennis_competitions = trading.betting.list_competitions(
        filter=betfairlightweight.filters.market_filter(event_type_ids=['2'])
    )
    return [
        comp.competition.id for comp in tennis_competitions
        if any(keyword in comp.competition.name for keyword in target_keywords)
    ]

def get_live_match_odds(trading: betfairlightweight.APIClient, competition_ids: List[str]) -> Tuple[List[MarketCatalogue], Dict[str, MarketBook]]:
    """
    Fetches market catalogues and books, retrying on failure with exponential backoff.

    Args:
        trading (betfairlightweight.APIClient): The authenticated API client.
        competition_ids (List[str]): A list of competition IDs to fetch markets for.

    Returns:
        Tuple[List[MarketCatalogue], Dict[str, MarketBook]]: A tuple containing a list
        of market catalogues and a dictionary mapping market IDs to market books.
    """
    # --- ADDED: Retry logic for fetching market data ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            market_catalogues = trading.betting.list_market_catalogue(
                filter=betfairlightweight.filters.market_filter(
                    competition_ids=competition_ids,
                    market_type_codes=['MATCH_ODDS']
                ),
                max_results=200,
                market_projection=['EVENT', 'RUNNER_METADATA', 'COMPETITION', 'DESCRIPTION']
            )

            if not market_catalogues:
                return [], {}

            market_ids = [market.market_id for market in market_catalogues]
            market_books = trading.betting.list_market_book(
                market_ids=market_ids,
                price_projection=betfairlightweight.filters.price_projection(price_data=['EX_BEST_OFFERS'])
            )
            market_book_lookup = {mb.market_id: mb for mb in market_books}
            return market_catalogues, market_book_lookup

        except APIError as e:
            error_string = str(e)
            # Don't retry for "No Markets Found" errors, as that is expected behavior
            if "DSC-0018" in error_string or "NO_MARKETS" in error_string:
                log_info("No active match odds markets found for the targeted competitions at this time.")
                return [], {}
            
            log_warning(f"API call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                log_info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                log_error("❌ All API attempts failed. Returning empty data for this run.")
                return [], {} # Return empty data to allow the pipeline to continue safely