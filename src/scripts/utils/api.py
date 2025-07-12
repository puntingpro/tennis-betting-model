# src/scripts/utils/api.py

import os
import betfairlightweight
from betfairlightweight import APIClient

def login_to_betfair(config: dict) -> APIClient:
    """
    Logs in to the Betfair API using username/password + app key only.
    Bypasses any certificate-based login by setting certs=None.
    """
    trading: APIClient = APIClient(
        username=os.getenv("BF_USER"),
        password=os.getenv("BF_PASS"),
        app_key=os.getenv("BF_APP_KEY"),
        certs=None               # <-- disable cert-based login
    )
    trading.login()
    return trading
