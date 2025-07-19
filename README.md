# 🎾 PuntingPro v2: Automated Betfair Tennis Trading Bot

[![CI Status](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml/badge.svg)](https://github.com/puntingpro/P1v2/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/puntingpro/P1v2/branch/main/graph/badge.svg)](https://codecov.io/gh/puntingpro/P1v2)
[![License](https://img.shields.io/github/license/puntingpro/P1v2.svg)](https://github.com/puntingpro/P1v2/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## 📈 Overview

PuntingPro v2 is an automated trading bot designed to identify and execute value bets on the Betfair Tennis exchange. The system leverages historical match data, advanced feature engineering, and a machine learning model (XGBoost) to predict match outcomes and find profitable trading opportunities in real-time.

This project is built for automation, using the official Betfair Stream API via the flumine Python framework to process market data and place bets with high speed and efficiency.

---

## ✨ Key Features

- **Data-Driven Predictions:** Uses a machine learning model trained on comprehensive historical tennis data to calculate match probabilities.
- **Automated Bet Execution:** Connects directly to the Betfair Exchange Stream API to monitor markets and place bets automatically without manual intervention.
- **Value Betting Logic:** Implements a core strategy to only place bets when the model's calculated odds are more favorable than the live market odds.
- **Robust CI/CD Pipeline:** Integrated with GitHub Actions for automated linting, type-checking, testing, and code coverage reporting to ensure code quality and stability.
- **Structured Project Layout:** Follows modern Python best practices with a `src` layout, virtual environments, and a clear, modular structure.

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.11
- **ML Model:** XGBoost
- **Betfair API Integration:** Flumine
- **Data Manipulation:** Pandas, Pandera
- **Hyperparameter Tuning:** Optuna
- **Testing:** Pytest
- **CI/CD:** GitHub Actions
- **Code Quality:** Black, Ruff, MyPy

---

## 🚀 Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/puntingpro/P1v2.git
cd P1v2
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\Activate.ps1
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
Install the project and all its dependencies in editable mode.
```bash
pip install -e .
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add:
```env
BETFAIR_USERNAME="your_betfair_username"
BETFAIR_PASSWORD="your_betfair_password"
BETFAIR_APP_KEY="your_live_app_key_from_betfair"
```
Place your Betfair API certificates (`client-2048.key` and `client-2048.crt`) in a `certs/` directory at the project root.

---

## ⚙️ Usage

Run the main pipeline:
```bash
python main.py
```
This will initialize the connection to the Betfair Stream API, start listening for market data, and place bets when value opportunities are identified.

---

## 📊 Project Status

The project is currently in **active development**. The core CI/CD pipeline is stable, and the foundational model training scripts are in place.

**Next Steps:**
- Integrate Official Betfair Data: Refactor data processing scripts to use official Betfair Data Portal.
- Upgrade to Flumine: Rearchitect the core betting logic for real-time stream processing and bet execution.
- Expand Test Coverage: Write comprehensive tests for the new Flumine-based logic and data modules.
