# 🎾 **tennis-betting-model: Automated Betfair Tennis Trading Bot**

---

## 📈 Overview

**tennis-betting-model** is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system uses Betfair's historical CSV Summary Files as the source for all core pricing and odds information and enriches it with historical match logs to ensure data completeness. The bot's strategy is designed to be compliant with Australian regulations, placing bets pre-play with a "Keep" persistence type, which allows the bet to remain active after the market goes in-play.

---

## 🛠️ Technology Stack

- **Programming Language**: Python
- **ML Model**: XGBoost
- **Data Manipulation**: Pandas
- **Hyperparameter Tuning**: Optuna
- **Dashboarding**: Streamlit

---

## 📂 Project Structure

The project is structured to centralize all application logic within the `src/` directory for better maintainability.

```text
└── tennis-betting-model/
    ├── .github/              # CI/CD workflows
    ├── data/
    │   ├── raw/              # Raw Betfair CSV summary files and historical CSVs
    │   ├── processed/        # Cleaned, consolidated data files
    │   └── analysis/         # Backtest results and outputs
    ├── models/               # Saved model files (.joblib)
    ├── src/
    │   └── tennis_betting_model/
    │       ├── analysis/       # Backtesting and profitability analysis
    │       ├── builders/       # Feature engineering and data preparation
    │       ├── dashboard/      # Streamlit dashboard application
    │       ├── modeling/       # Model training and evaluation
    │       ├── pipeline/       # Live trading automation
    │       └── utils/          # Shared utilities (config, logger, etc.)
    ├── tests/                # Unit and integration tests
    ├── config.yaml           # Project configuration
    ├── main.py               # Main CLI entrypoint
    └── README.md             # This file
```

---

## 🚀 Full Project Pipeline (Commands)

This is the complete sequence of commands to run the entire pipeline from raw data to analysis. These should be run from the project's root directory.

### Step 1: Prepare Raw Data

Prepares raw data assets including player attributes, historical rankings, and combines Betfair CSV files into a raw odds file.

```bash
python main.py prepare-data
```

### Step 2: Create Player Map (One-Time Setup)

Generates `player_mapping.csv` by matching players between Betfair and historical datasets. Manually review the file afterward.

```bash
python main.py create-player-map
```

### Step 3: Build All Features

Enriches raw odds data, creates match logs, calculates Elo ratings, and engineers features.

```bash
python main.py build
```

### Step 4: Train the Model

Trains a new XGBoost model on the engineered features.

```bash
python main.py model
```

### Step 5: Run a Realistic Backtest

Executes a backtest using the trained model and historical odds.

```bash
python main.py backtest realistic
```

### Step 6: Analyze Results

Explore backtest results using various analysis commands:

```bash
# Summarize tournament performance
python main.py analysis summarize-tournaments

# Plot the leaderboard (opens a plot window)
python main.py analysis plot-leaderboard --show-plot
```

### Step 7: 📊 Launch the Interactive Dashboard

Launch a Streamlit dashboard for interactive result analysis.

```bash
python main.py dashboard
```

### Automation

To run the live betting pipeline on a schedule (as defined in GitHub Actions), use:

```bash
python main.py automate
