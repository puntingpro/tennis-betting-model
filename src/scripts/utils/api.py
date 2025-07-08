# src/scripts/utils/api.py
import os
import betfairlightweight
from betfairlightweight.exceptions import APIError
from typing import List, Tuple
from .logger import log_info

def login_to_betfair(config: dict) -> betfairlightweight.APIClient:
    """Logs in to the Betfair API."""
    trading = betfairlightweight.APIClient(
        username=os.getenv('BF_USER'),
        password=os.getenv('BF_PASS'),
        app_key=os.getenv('BF_APP_KEY')
    )
    trading.login_interactive()
    return trading

def get_tennis_competitions(trading: betfairlightweight.APIClient, target_keywords: List[str]) -> List[str]:
    """Fetches and filters tennis competitions by keywords."""
    tennis_competitions = trading.betting.list_competitions(
        filter=betfairlightweight.filters.market_filter(event_type_ids=['2'])
    )
    return [
        comp.competition.id for comp in tennis_competitions
        if any(keyword in comp.competition.name for keyword in target_keywords)
    ]

def get_live_match_odds(trading: betfairlightweight.APIClient, competition_ids: List[str]) -> Tuple[list, dict]:
    """Fetches market catalogues and books for live matches, handling API errors gracefully."""
    try:
        market_catalogues = trading.betting.list_market_catalogue(
            filter=betfairlightweight.filters.market_filter(
                competition_ids=competition_ids,
                market_type_codes=['MATCH_ODDS']
            ),
            max_results=200,
            market_projection=['EVENT', 'RUNNER_METADATA', 'COMPETITION', 'DESCRIPTION']
        )
    except APIError as e:
        # --- FINAL FIX: Check the string representation of the error ---
        if "DSC-0018" in str(e):
            log_info("No active match odds markets found for the targeted competitions at this time.")
            return [], {}
        # --- END FINAL FIX ---
        else:
            # Re-raise other, more critical API errors
            raise

    market_ids = [market.market_id for market in market_catalogues]
    if not market_ids:
        return [], {}

    market_books = trading.betting.list_market_book(
        market_ids=market_ids,
        price_projection=betfairlightweight.filters.price_projection(price_data=['EX_BEST_OFFERS'])
    )
    market_book_lookup = {mb.market_id: mb for mb in market_books}
    return market_catalogues, market_book_lookup