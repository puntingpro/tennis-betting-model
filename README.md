# 🎾 PuntingPro: Tennis Value Betting Pipeline (v1.2)
A modern, API-driven, and modular pipeline for finding and simulating value bets in ATP and WTA tennis.

[![CI Status](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml/badge.svg)](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml) [![Codecov](https://codecov.io/gh/puntingpro/P1v2/graph/badge.svg)](https://codecov.io/gh/puntingpro/P1v2)

### **Overview**
This project uses a two-stage process that combines historical data with a live API connection to find profitable betting opportunities. It is designed to be robust, configurable, and easy to use from the command line.

1.  **Offline Feature Engineering**: Uses historical match data to build a rich feature set for every player, including stats like win percentages, head-to-head records, and Elo ratings.
2.  **Live Value-Finding Pipeline**: Connects to the Betfair API to fetch real-time match odds, enriches this data with the pre-calculated features, and uses a predictive XGBoost model to identify and alert on value bets.

### ✨ Features
* **End-to-End Workflow:** From raw data consolidation to live value detection.
* **Advanced Feature Engineering:** Includes Elo ratings, rolling form, and surface-specific win percentages.
* **Hyperparameter Optimization:** Uses Optuna to find the best parameters for the prediction model.
* **Sophisticated Backtesting:**
    * Simulates historical betting based on the model's predictions.
    * Provides detailed, categorized summaries of performance by tournament type.
    * Includes crucial risk metrics like ROI and filters for statistical significance.
* **Live Automation:** Can run continuously on a schedule to find and alert on new betting opportunities.
* **Interactive Dashboard:** A Streamlit dashboard to visualize model performance and backtest results.
* **High-Quality Codebase:** Adheres to modern Python standards with type hinting, automated formatting (`black`), and linting (`ruff`).

---

### **Project Structure**
```
P1v2/
├── main.py               # CLI entrypoint for all operations
├── config.yaml           # Primary project configuration file
├── requirements.txt      # Project dependencies
├── README.md             # This file
├── models/               # Stores trained model artifacts
├── data/                 # For all raw, processed, and analysis data
├── src/                  # All Python source code
│   ├── scripts/
│   │   ├── analysis/
│   │   ├── builders/
│   │   ├── dashboard/
│   │   ├── modeling/
│   │   ├── pipeline/
│   │   └── utils/
│   └── ...
└── tests/                # Unit and integration tests
```

---

### **Setup and Installation**

**1. Clone the Repository**
```bash
git clone <repository_url>
cd P1v2
```

**2. Configure Git (Important for Windows Users)**
To prevent cross-platform line-ending issues, run this one-time command:
```bash
git config --global core.autocrlf true
```

**3. Create and Activate Virtual Environment**
```bash
# Create the environment
python -m venv .venv

# Activate the environment
# On Windows (PowerShell):
. .\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

**4. Set Up a Command Alias (Optional, Recommended for PowerShell)**
To avoid typing the full activation command every time, you can create a simple one-word alias.
* Open your PowerShell profile: `notepad $PROFILE`
* Add the following lines and save the file:
    ```powershell
    # Custom Alias to Activate the P1v2 Virtual Environment
    function Activate-P1v2-Venv {
        . C:\path\to\your\P1v2\.venv\Scripts\Activate.ps1
    }
    Set-Alias -Name p1 -Value Activate-P1v2-Venv
    ```
* Restart your terminal. You can now simply type `p1` to activate the environment.

**5. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**6. Configure a `config.yaml` file** and set your Betfair credentials and SSL certificates as instructed by Betfair.

---

### **Workflow & Usage**

All project operations are run through `main.py`. Here is the recommended workflow:

**Step 1: Build the Model (One-Time Setup)**
These commands process your raw data and train the machine learning model.
```bash
# 1. Consolidate all raw data files
python main.py consolidate

# 2. Build the advanced player features and Elo ratings
python main.py build

# 3. Train the predictive model
python main.py model
```

**Step 2: Analyze Your Strategy**
Before betting, run a historical backtest to understand your model's performance.
```bash
# 1. Run the backtest to generate results
python main.py backtest

# 2. Summarize the results by tournament category
# Use --min-bets to ensure statistical significance
python main.py analysis summarize-tournaments --min-bets 100
```

**Step 3: Run the Live Pipeline**
Once your Betfair Live Key is active and you have refined your `config.yaml`, you can start finding bets.
```bash
# To run a single, manual check for value bets (no bets placed):
python main.py pipeline --dry-run

# To start the fully automated service that runs every 15 minutes:
python main.py automate
```

---

### **Development**
This project uses the following tools to maintain code quality. The CI pipeline will automatically run these checks on every commit.

* **`black`**: For consistent, automated code formatting. Run `black .` to format the code.
* **`ruff`**: For fast, efficient linting to catch common errors.
* **`mypy`**: For static type checking to prevent type-related bugs.
