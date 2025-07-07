# src/scripts/pipeline/fetch_live_matches.py

import os
import betfairlightweight
import pandas as pd

def fetch_live_matches():
    """
    Connects to the Betfair API and fetches all available tennis matches.
    """
    # --- 1. LOGIN ---
    # Load credentials from environment variables
    my_username = os.getenv('BF_USER')
    my_password = os.getenv('BF_PASS')
    my_app_key = os.getenv('BF_APP_KEY')

    if not all([my_username, my_password, my_app_key]):
        raise ValueError("Please set BF_USER, BF_PASS, and BF_APP_KEY environment variables.")

    # Create a trading instance (no certs path needed)
    trading = betfairlightweight.APIClient(
        username=my_username,
        password=my_password,
        app_key=my_app_key,
    )

    # Use interactive login to bypass certificate requirements
    print("Logging in to Betfair using interactive method...")
    trading.login_interactive()
    print("Login successful.")

    # --- 2. FIND TENNIS EVENTS ---
    # Create a filter to find the event type for Tennis
    tennis_event_filter = betfairlightweight.filters.market_filter(
        text_query='Tennis'
    )

    # Get a list of all event types (competitions) for Tennis
    print("Finding tennis event types...")
    tennis_event_types = trading.betting.list_event_types(
        filter=tennis_event_filter
    )

    if not tennis_event_types:
        print("Could not find any event types for Tennis. Exiting.")
        return

    # Extract the event type ID for Tennis (usually '2')
    tennis_event_type_id = tennis_event_types[0].event_type.id
    print(f"Found Tennis Event Type ID: {tennis_event_type_id}")

    # --- 3. GET LIVE MATCHES (MARKET CATALOGUES) ---
    # Create a filter for match odds in Tennis
    market_filter = betfairlightweight.filters.market_filter(
        event_type_ids=[tennis_event_type_id],
        market_type_codes=['MATCH_ODDS']
    )

    # Get all market catalogues (i.e., matches)
    print("Fetching live tennis matches...")
    market_catalogues = trading.betting.list_market_catalogue(
        filter=market_filter,
        max_results=200,  # Get up to 200 matches
        market_projection=['EVENT', 'COMPETITION', 'RUNNER_DESCRIPTION']
    )

    if not market_catalogues:
        print("No live tennis matches found.")
        return

    # --- 4. DISPLAY RESULTS ---
    match_list = []
    for market in market_catalogues:
        match_list.append({
            'Competition': market.competition.name,
            'Match': market.event.name,
            'Market ID': market.market_id,
            'Start Time': market.market_start_time,
            'Player 1': market.runners[0].runner_name,
            'Player 2': market.runners[1].runner_name,
        })

    # Create a pandas DataFrame for clean printing
    matches_df = pd.DataFrame(match_list)

    print("\n--- Available Tennis Matches ---")
    print(matches_df.to_string())
    print(f"\nFound {len(matches_df)} matches.")

    # Optional: Logout
    trading.logout()
    print("\nLogged out.")


if __name__ == "__main__":
    fetch_live_matches()