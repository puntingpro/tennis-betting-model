<p align="center">
  <img src="https://github.com/puntingpro/P1v2/actions/workflows/ci.yml/badge.svg" />
  <a href="https://codecov.io/gh/puntingpro/P1v2" > 
    <img src="https://codecov.io/gh/puntingpro/P1v2/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE"/> 
  </a>
</p>

# 🎾 Tennis Value Betting Pipeline (P1v2)

*A modern, modular, and fully tested pipeline for finding and simulating value bets in ATP and WTA tennis.*

---

## 🚀 Getting Started

To set up the project for the first time, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd P1v2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:** The project uses a locked `requirements.txt` for reproducible builds.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore Commands:** The project uses a single entrypoint. Discover all commands:
    ```bash
    python main.py --help
    ```

---

## ▶️ Quickstart

1.  **Run the full pipeline for a single configuration:**
    ```bash
    python main.py pipeline --config configs/pipeline_run.yaml
    ```

2.  **Train a model on the pipeline's output:**
    ```bash
    python main.py model train-filter \
      --input_glob "data/processed/*_value_bets.csv" \
      --output_model models/ev_filter.joblib
    ```

3.  **Analyze the results:**
    ```bash
    python main.py analysis summarize-matches \
        --value_bets_glob "data/processed/*_value_bets.csv" \
        --output_csv "data/plots/match_summary.csv"
    ```

---

## 📄 Canonical Data Columns (Pipeline Contract)

| Column             | Type      | Description                                                         |
| ------------------ | --------- | ------------------------------------------------------------------- |
| `match_id`         | str       | Unique identifier for the match, derived from Betfair's `market_id`.|
| `market_id`        | str       | Betfair market identifier.                                          |
| `player_1`         | str       | Standardized name of player 1 (for modeling).                       |
| `player_2`         | str       | Standardized name of player 2.                                      |
| `odds`             | float     | Decimal odds for the bet.                                           |
| `predicted_prob`   | float     | Model-predicted probability the bet wins (0 to 1).                  |
| `expected_value`   | float     | Calculated EV (`(predicted_prob * (odds - 1)) - (1 - predicted_prob)`). |
| `winner`           | int       | 1 if the bettor on this row won, 0 otherwise (ground truth).        |
| `confidence_score` | float     | (optional) Model confidence in its prediction.                      |
| `kelly_fraction`   | float     | (optional) Fraction of bankroll to bet, per Kelly formula.          |
| `timestamp`        | int/float | (optional) Unix or Betfair timestamp of the last traded price.      |

---

## 📂 Project Structure

project_root/  
├── main.py                      # Unified CLI entrypoint  
├── configs/                     # YAML configs for pipelines and tournaments  
├── src/scripts/analysis/        # Analysis scripts (EV plots, summaries)  
├── src/scripts/builders/        # Match-building and orchestration  
├── src/scripts/pipeline/        # Core pipeline stages (features, predict, detect, simulate)  
├── src/scripts/utils/           # Shared utilities (CLI, logging, config, normalization)  
├── tests/                       # Unit & integration tests  
├── README.md                    # This file  
└── requirements.in              # Source for dependencies

---

## 🔄 Batch Pipeline Runs

To process **all tournaments** listed in your tournaments YAML in one go, add the `--batch` flag:

```bash
python main.py pipeline --config configs/tournaments_2024.yaml --batch
```

---

## 📊 Model Reproducibility

Each model training saves, alongside the .joblib model file, a metadata .json containing:

- Timestamp & git commit hash  
- Model type and feature list  
- EV threshold, training-row count, and other parameters  

---

## ✅ Tests

We keep tests in the top-level `tests/` directory. To run:

```bash
pytest -v
```
