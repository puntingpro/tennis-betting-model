# 🎾 **tennis-betting-model: Automated Betfair Tennis Trading Bot**
---
## 📈 Overview
**tennis-betting-model** is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system uses Betfair's historical CSV Summary Files as the source for all core pricing and odds information and enriches it with historical match logs to ensure data completeness. The bot's strategy is designed to be compliant with Australian regulations, placing bets pre-play with a "Keep" persistence type, which allows the bet to remain active after the market goes in-play.

---
## 🛠️ Technology Stack
- **Programming Language**: Python
- **ML Model**: LightGBM
- **Real-Time Trading**: Flumine (Betfair Stream API wrapper)
- **Data Manipulation**: Pandas
- **Hyperparameter Tuning**: Optuna
- **Dashboarding**: Streamlit

---
## 📂 Project Structure
(No changes to this section)

---
### IMPROVEMENT: Revised Project Pipeline Commands
This is the complete sequence of commands for the recommended workflow. These should be run from the project's root directory.

### Part 1: Data Processing & Model Training (Scheduled or Manual)
This workflow prepares all data, builds features, trains the model, and runs a backtest. It should be run periodically (e.g., weekly) to keep the model fresh. The provided `.github/workflows/build_and_backtest.yml` automates this.

**Step 1: Prepare Raw Data**
```bash
python main.py prepare-data
```

**Step 2: Create Player Map (One-Time Setup & Review)**
```bash
python main.py create-player-map
```
# Visually review and correct mappings if necessary:
```bash
python main.py analysis review-mappings
```

**Step 3: Build All Features**
```bash
python main.py build
```

**Step 4: Train the Model**
```bash
python main.py model
```

**Step 5: Run a Realistic Backtest & Analyze Results**
```bash
python main.py backtest realistic
```bash
python main.py analysis summarize-tournaments
```bash
python main.py analysis plot-leaderboard --show-plot
```

### Part 2: Analysis & Live Trading

**Step 6: 📊 Launch the Interactive Dashboard**
Launch a Streamlit dashboard to interactively analyze the latest backtest results.
```bash
python main.py dashboard
```

**Step 7: ⚡ Run the Live Trading Bot**
Run the live betting bot using the real-time Betfair Stream API. This should be run as a long-running process on a server/VPS. It's highly recommended to use --dry-run first.
```bash
# Run in dry-run mode (no real money)
python main.py stream --dry-run
```bash
# Run in LIVE mode (places real bets)
python main.py stream
