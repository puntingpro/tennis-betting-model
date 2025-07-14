# 🎾 Tennis Value Betting Pipeline (P1v2)
A modern, API-driven, and modular pipeline for finding and simulating value bets in ATP and WTA tennis.

# --- MODIFIED: Removed placeholder token ---
[![CI Status](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml/badge.svg)](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml) [![Codecov](https://codecov.io/gh/puntingpro/P1v2/graph/badge.svg)](https://codecov.io/gh/puntingpro/P1v2)
# --- END MODIFICATION ---

### **Overview**
This project uses a two-stage process that combines historical data with a live API connection to find profitable betting opportunities.

1.  **Offline Feature Engineering**: Uses historical data to build a rich feature set for every player, including stats like win percentages and recent form.
2.  **Live Value-Finding Pipeline**: Connects to the Betfair API to fetch real-time match odds, enriches this data with the pre-calculated features, and uses a predictive model to identify value bets.

### **Quickstart Guide**

1.  **Initial Setup**
    ```bash
    git clone <repository_url>
    cd P1v2
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
    Create a `config.yaml` file (you can copy the example) and set your Betfair credentials as environment variables (`BF_USER`, `BF_PASS`, `BF_APP_KEY`).

2.  **Build Historical Features**
    *This command processes raw data to create the feature library for the model. It only needs to be run once.*
    ```bash
    python main.py --config config.yaml build
    ```

3.  **Train the Predictive Model**
    *This command uses the features to train the model and find the best hyperparameters. This can take a few hours but only needs to be run once.*
    ```bash
    python main.py --config config.yaml model
    ```

4.  **Run the Live Pipeline**
    *This is the main command you will use. It scans for live matches and identifies value bets in real-time.*
    ```bash
    python main.py --config config.yaml pipeline
    ```

5.  **(Optional) Run Analysis**
    *After running a backtest, you can analyze and visualize the results.*
    ```bash
    # Run the backtest
    python main.py --config config.yaml backtest

    # Summarize and plot the results
    python main.py --config config.yaml analysis summarize-tournaments
    python main.py --config config.yaml analysis plot-leaderboard
    ```

### **Project Structure**
P1v2/
├── main.py               # CLI entrypoint
├── config.yaml           # Project configuration
├── models/               # Trained model artifacts
├── data/                 # Raw, processed & analysis data
├── src/                  # All Python source code
├── tests/                # Unit & integration tests
├── README.md             # This file
└── requirements.txt      # Project dependencies
