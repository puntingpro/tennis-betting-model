🎾 tennis-betting-model: Automated Betfair Tennis Trading Bot
📈 Overview

tennis-betting-model is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system uses Betfair's historical CSV Summary Files as the source for all core pricing and odds information and enriches it with historical match logs to ensure data completeness. The bot's strategy is designed to be compliant with Australian regulations, placing bets pre-play with a "Keep" persistence type, which allows the bet to remain active after the market goes in-play.

---
🛠️ Technology Stack

* **Programming Language**: Python
* **ML Model**: XGBoost
* **Data Manipulation**: Pandas
* **Hyperparameter Tuning**: Optuna
* **Data Extraction**: Pandas

---
📂 Project Structure
The project is structured to centralize all application logic within the `src/` directory for better maintainability.
└── tennis-betting-model/
    ├── .github/              # CI/CD workflows
    ├── data/
    │   ├── raw/              # Raw Betfair CSV summary files and historical CSVs
    │   ├── processed/        # Cleaned, consolidated data files
    │   └── analysis/         # Backtest results and outputs
    ├── models/               # Saved model files (.joblib)
    ├── src/
    │   └── tennis_betting_model/ # Main application source code
    │       ├── analysis/       # Backtesting and profitability analysis
    │       ├── builders/       # Feature engineering and data preparation
    │       ├── modeling/       # Model training and evaluation
    │       ├── pipeline/       # Live trading automation
    │       └── utils/          # Shared utilities (config, logger, etc.)
    ├── tests/                # Unit and integration tests
    ├── config.yaml           # Project configuration
    ├── main.py               # Main CLI entrypoint
    └── README.md             # This file

---
🚀 Full Project Pipeline (Commands)
This is the complete sequence of commands to run the entire pipeline from raw data to a realistic backtest. These should be run from the project's root directory.

### Step 1: Prepare Raw Data
This command prepares the most raw data assets. It consolidates player attributes, historical rankings, and combines all Betfair CSV summary files into a single raw odds file.
```bash
python main.py prepare-data
Step 2: Create Player Map (One-Time Setup)
This helper script generates a player_mapping.csv file by matching players between the Betfair and historical datasets. You should run this once after preparing new data and manually review the output file for accuracy.

Bash

python main.py create-player-map
Step 3: Build All Features
This command builds all derived data assets. It enriches the raw odds data, creates the definitive match log, calculates surface-specific Elo ratings, and engineers all features for the model.

Bash

python main.py build
Step 4: Train the Model
Trains a new XGBoost model on the features generated in the previous step.

Bash

python main.py model
Step 5: Run a Realistic Backtest
Executes a backtest using the new model and real historical Betfair odds to gauge performance.

Bash

python main.py backtest realistic
