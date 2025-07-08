# 🎾 Tennis Value Betting Pipeline (P1v2)
A modern, API-driven, and modular pipeline for finding and simulating value bets in ATP and WTA tennis.

[![CI Status](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml/badge.svg)](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml) [![Codecov](https://codecov.io/gh/puntingpro/P1v2/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/puntingpro/P1v2)

🚀 **Architecture: A Hybrid Approach**

This project uses a two-stage process that combines historical data and live market odds via the Betfair API, avoiding reconciliation issues of file‑only systems.

**Offline Feature Engineering**  
Uses Sackmann Tennis Data to build features like `rank_diff`, `surface_win_percentage`, and `recent_form`. Runs infrequently to preprocess years of data into `data/processed/`.

**Live Value-Finding Pipeline**  
Fetches live odds from Betfair, enriches with historical features, runs an XGBoost model, and flags matches where model probability > implied probability (value bets).

▶️ **Quickstart Guide**

1. **Initial Setup**
   ```bash
   git clone <repository_url>
   cd P1v2
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Store `BF_USER`, `BF_PASS`, and `BF_APP_KEY` in your environment.

2. **Generate Historical Features**
   ```bash
   python main.py build
   ```

3. **Train the Predictive Model**
   ```bash
   python main.py model --input_glob "data/processed/*_features.csv" \
     --output_model "models/advanced_xgb_model.joblib"
   ```

4. **Run the Live Pipeline**
   ```bash
   python main.py pipeline
   ```

5. **Backtest Your Strategy**
   ```bash
   python main.py backtest \
     --model_path "models/advanced_xgb_model.joblib" \
     --features_csv "data/processed/all_advanced_features.csv" \
     --output_csv "data/analysis/backtest_results.csv"
   ```

📂 **Project Structure**
```
P1v2/
├── main.py                   # CLI entrypoint
├── models/                   # Trained model artifacts
├── data/                     # Raw, processed & analysis data
│   ├── processed/
│   └── analysis/
├── src/scripts/              # builders, modeling, pipeline, utils
├── tests/                    # Unit & integration tests
├── README.md                 # This file
└── requirements.txt          # Dependencies
```
