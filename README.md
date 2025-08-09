# 🎾 **tennis-betting-model: Automated Betfair Tennis Trading Bot**

## 📈 Overview
**tennis-betting-model** is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system uses historical data to build features, train a predictive model, and find profitable betting opportunities. The bot's strategy is designed to place bets pre-play with a "Keep" persistence type, allowing the bet to remain active after the market goes in-play.
---

## 🛠️ Technology Stack
-   **Programming Language**: Python
-   **ML Model**: LightGBM
-   **Real-Time Trading**: Flumine (Betfair Stream API wrapper)
-   **Data Manipulation**: Pandas
-   **Hyperparameter Tuning**: Optuna
-   **Dashboarding**: Streamlit

---

## 📂 Project Structure
```
.
├── .github/workflows/        # GitHub Actions for CI, building, and streaming
├── data/                       # Project data (raw, processed, analysis)
├── models/                     # Trained machine learning models
├── src/tennis_betting_model/ # Main source code
│   ├── analysis/             # Backtesting, analysis, and plotting scripts
│   ├── builders/             # Scripts for data preparation, feature engineering
│   ├── modeling/             # Model training and evaluation code
│   ├── pipeline/             # Real-time Flumine trading logic
│   └── utils/                # Utility functions (logging, config, etc.)
├── tests/                      # Unit and integration tests
├── main.py                     # Main CLI entry point for all commands
└── conf/config.yaml            # Single, consolidated configuration file
```

---

## ⚙️ Project Pipeline Commands
This is the complete sequence of commands for the recommended workflow. These should be run from the project's root directory.

### Part 1: Initial Setup & Data Processing
This workflow prepares all raw data sources, consolidates them, and creates the necessary mappings.

**Step 1: Prepare Raw Data**
Consolidates raw player data, historical rankings, and Betfair summary odds files into processed CSVs.
```bash
python main.py command=prepare-data
```

**Step 2: Create Player Map**
Generates a mapping file to link Betfair player IDs with historical player IDs.
```bash
python main.py command=create-player-map
```

### Part 2: Model Training & Backtesting
This workflow builds the features, trains the model, and runs a historical backtest.

**Step 3: Build All Features & Backtest Data**
Runs the full data enrichment pipeline, including creating the match log, calculating Elo ratings, building features, and preparing the data for a realistic backtest.
```bash
python main.py command=build
```

**Step 4: Train the Model**
Trains the LightGBM model using the features generated in the previous step.
```bash
python main.py command=model
```

**Step 5: Run Backtest & Generate Analysis**
Runs a historical backtest using the trained model and realistic market odds, then generates analysis reports.
```bash
python main.py command=backtest mode=realistic
python main.py command=analysis/summarize-tournaments
python main.py command=analysis/plot-leaderboard
```

### Part 3: Live Trading & Analysis

**Step 6: 📊 Launch the Interactive Dashboard**
Launch a Streamlit dashboard to interactively analyze the latest backtest results and simulate staking strategies.
```bash
python main.py command=dashboard
```

**Step 7: ⚡ Run the Live Trading Bot**
Run the live betting bot using the real-time Betfair Stream API. It is highly recommended to run in dry-run mode first.
```bash
# Run in dry-run mode (no real money)
python main.py command=stream dry_run=true
```
```bash
# Run in LIVE mode (places real bets)
python main.py command=stream
```
