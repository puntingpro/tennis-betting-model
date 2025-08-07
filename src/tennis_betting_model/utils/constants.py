# --- Betting Strategy & Backtesting ---
DEFAULT_EV_THRESHOLD = 0.1  # 10% edge
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # 50% probability
DEFAULT_MAX_ODDS = 10.0
DEFAULT_MAX_MARGIN = 1.05  # Corresponds to a 5% bookmaker margin
BOOKMAKER_MARGIN = 1.05  # Represents a 5% margin for odds simulation
BACKTEST_MAX_ODDS = 50.0

# --- Data Processing & Feature Engineering ---
ELO_K_FACTOR = 32
ELO_INITIAL_RATING = 1500
DEFAULT_PLAYER_RANK = 500  # Moved from common.py

# --- Simulation Defaults ---
DEFAULT_INITIAL_BANKROLL = 1000.0
