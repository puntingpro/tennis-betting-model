# 🎾 tennis-betting-model: Automated Betfair Tennis Trading Bot

## 📈 Overview
`tennis-betting-model` is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system uses Betfair's historical PRO-level data as the source for all core pricing and odds information, and enriches it with historical match logs to ensure data completeness. The bot's strategy is designed to be compliant with Australian regulations, placing bets pre-play with a "Keep" persistence type, which allows the bet to remain active after the market goes in-play.

## 🛠️ Technology Stack
* **Programming Language**: Python
* **ML Model**: XGBoost
* **Data Manipulation**: Pandas
* **Hyperparameter Tuning**: Optuna
* **Data Extraction**: `tarfile`, `bz2`, `orjson`

## 📂 Project Structure
The project has been refactored to centralize all core application logic within the `src/` directory.
└── tennis-betting-model/
├── data/
│   ├── raw/              # Raw Betfair .tar files and historical CSVs
│   ├── processed/        # Cleaned, consolidated data files
│   └── analysis/         # Backtest results and outputs
├── models/               # Saved model files (.joblib)
├── src/
│   └── tennis_betting_model/ # Main application source code
│       ├── analysis/      # Backtesting and profitability analysis
│       ├── builders/       # Feature engineering and data building
│       ├── modeling/       # Model training and evaluation
│       ├── pipeline/       # Live trading automation
│       └── utils/          # Shared utilities (config, logger, etc.)
├── .github/              # CI/CD workflows
├── create_mapping_file.py # Helper script to map player IDs
├── main.py               # Main CLI entrypoint
└── config.yaml           # Project configuration


## 🚀 Full Project Pipeline (Commands)
This is the complete, streamlined sequence of commands to run the entire pipeline from raw data to a realistic backtest. These should be run from the project's root directory.

**Step 0: Create Player Map (One-Time Setup)**
This helper script generates a `player_mapping.csv` file by matching players between the Betfair and historical datasets. You must run this once and manually review the output file for accuracy.
```bash
python scripts/create_mapping_file.py
Step 1: Prepare All Data
This single command runs the entire data preparation pipeline: it extracts data from the raw .tar files, enriches it using the player map, and creates the final match log.

Bash

python main.py prepare-data
Step 2: Build All Features & Elo Ratings
This command reads the prepared data to calculate Elo ratings and engineer all features for the model.

Bash

python main.py build
Step 3: Train the Model
Trains a new XGBoost model on the features generated in the previous step.

Bash

python main.py model
Step 4: Run a Realistic Backtest
Executes a backtest using the new model and real historical Betfair odds to gauge performance.

Bash

python main.py backtest realistic
