# tests/analysis/test_summarize_value_bets_by_tournament.py

import pandas as pd
import pytest
from src.scripts.analysis.summarize_value_bets_by_tournament import (
    run_summarize_by_tournament,
)


@pytest.fixture
def sample_backtest_results() -> pd.DataFrame:
    """
    Creates a sample DataFrame of backtest results to test the summary logic.
    This fixture includes a key scenario that exposed the original bug:
    - A winning bet (winner=1) where the predicted probability was less than 0.5.
      The old logic would incorrectly mark this as a loss.
    """
    data = {
        "tourney_name": ["ATP Masters", "Grand Slam", "ATP Masters", "Grand Slam"],
        "predicted_prob": [0.8, 0.4, 0.7, 0.9],
        "winner": [1, 1, 0, 1],
        "odds": [1.5, 3.0, 2.0, 1.2],
    }
    return pd.DataFrame(data)


def test_summarize_by_tournament_profit_calculation(sample_backtest_results):
    """
    Tests that the profit and ROI calculations are correct, especially for
    winning bets where the predicted probability was less than 0.5.
    """
    summary_df = run_summarize_by_tournament(sample_backtest_results, min_bets=1)

    # --- Verification ---
    # 1. Grand Slam Category:
    #    - Bet 1: winner=1, odds=3.0. Profit = 3.0 - 1 = 2.0
    #    - Bet 2: winner=1, odds=1.2. Profit = 1.2 - 1 = 0.2
    #    - Total Bets: 2
    #    - Total Profit: 2.0 + 0.2 = 2.2
    #    - ROI: (2.2 / 2) * 100 = 110%
    gs_summary = summary_df[summary_df["tourney_category"] == "Grand Slam"].iloc[0]
    assert gs_summary["total_bets"] == 2
    assert pytest.approx(gs_summary["total_profit"]) == 2.2
    assert pytest.approx(gs_summary["roi"]) == 110.0

    # 2. Masters 1000 Category:
    #    - Bet 1: winner=1, odds=1.5. Profit = 1.5 - 1 = 0.5
    #    - Bet 2: winner=0, odds=2.0. Profit = -1.0
    #    - Total Bets: 2
    #    - Total Profit: 0.5 - 1.0 = -0.5
    #    - ROI: (-0.5 / 2) * 100 = -25%
    masters_summary = summary_df[summary_df["tourney_category"] == "Masters 1000"].iloc[
        0
    ]
    assert masters_summary["total_bets"] == 2
    assert pytest.approx(masters_summary["total_profit"]) == -0.5
    assert pytest.approx(masters_summary["roi"]) == -25.0
