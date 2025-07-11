# src/scripts/utils/api.py
import os
import betfairlightweight
from betfairlightweight.exceptions import APIError
from typing import List, Tuple
from .logger import log_info, log_warning

def login_to_betfair(config: dict) -> betfairlightweight.APIClient:
    """Logs in to the Betfair API using a non-interactive certificate login."""
    cert_path = os.getenv('BF_CERT_PATH', 'certs')
    os.makedirs(cert_path, exist_ok=True)
    
    cert_file = os.path.join(cert_path, 'cert.pem')
    key_file = os.path.join(cert_path, 'key.pem')

    # --- MODIFIED: Use .strip() to remove extra whitespace ---
    with open(cert_file, 'w') as f:
        f.write(os.environ['BF_CERT'].strip())
    with open(key_file, 'w') as f:
        f.write(os.environ['BF_KEY'].strip())
        
    trading = betfairlightweight.APIClient(
        username=os.getenv('BF_USER'),
        password=os.getenv('BF_PASS'),
        app_key=os.getenv('BF_APP_KEY'),
        certs=cert_path
    )
    trading.login()
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
        error_string = str(e)
        if "DSC-0018" in error_string or "NO_MARKETS" in error_string:
            log_info("No active match odds markets found for the targeted competitions at this time.")
        else:
            log_warning(f"An unexpected Betfair API error occurred: {error_string}")
        return [], {}

    market_ids = [market.market_id for market in market_catalogues]
    if not market_ids:
        return [], {}

    market_books = trading.betting.list_market_book(
        market_ids=market_ids,
        price_projection=betfairlightweight.filters.price_projection(price_data=['EX_BEST_OFFERS'])
    )
    market_book_lookup = {mb.market_id: mb for mb in market_books}
    return market_catalogues, market_book_lookup