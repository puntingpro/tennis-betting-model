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
└── config.yaml                 # Configuration file for paths, parameters, etc.
```

---

## ⚙️ Project Pipeline Commands
This is the complete sequence of commands for the recommended workflow. These should be run from the project's root directory. The entire build and backtest process is automated via the `.github/workflows/build_and_backtest.yml` workflow.

### Part 1: Initial Setup & Data Processing
This workflow prepares all raw data sources, consolidates them, and creates the necessary mappings.

**Step 1: Prepare Raw Data**
Consolidates raw player data, historical rankings, and Betfair summary odds files into processed CSVs.
```bash
python main.py prepare-data
```

**Step 2: Create Player Map (Review Recommended)**
Generates a mapping file to link Betfair player IDs with historical player IDs.
```bash
python main.py create-player-map
```
To visually review and correct any ambiguous mappings, launch the interactive tool:
```bash
python main.py analysis review-mappings
```

### Part 2: Model Training & Backtesting
This workflow builds the features, trains the model, and runs a historical backtest.

**Step 3: Build All Features & Backtest Data**
Runs the full data enrichment pipeline, including creating the match log, calculating Elo ratings, building features, and preparing the data for a realistic backtest.
```bash
python main.py build
```

**Step 4: Train the Model**
Trains the LightGBM model using the features generated in the previous step.
```bash
python main.py model
```

**Step 5: Run Backtest & Generate Analysis**
Runs a historical backtest using the trained model and realistic market odds. It then generates analysis reports on the results.
```bash
python main.py backtest realistic
python main.py analysis summarize-tournaments
python main.py analysis plot-leaderboard
```

### Part 3: Live Trading & Analysis

**Step 6: 📊 Launch the Interactive Dashboard**
Launch a Streamlit dashboard to interactively analyze the latest backtest results and simulate staking strategies.
```bash
python main.py dashboard
```

**Step 7: ⚡ Run the Live Trading Bot**
Run the live betting bot using the real-time Betfair Stream API. It is highly recommended to run in `--dry-run` mode first to ensure everything is working correctly without risking real money. This command is designed to be run as a long-running process on a server or VPS.
```bash
# Run in dry-run mode (no real money)
python main.py stream --dry-run
```bash
# Run in LIVE mode (places real bets)
python main.py stream
