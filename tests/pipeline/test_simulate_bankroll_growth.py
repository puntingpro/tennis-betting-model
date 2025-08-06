# tests/pipeline/test_simulate_bankroll_growth.py

import pandas as pd
import pytest
from tennis_betting_model.pipeline.simulate_bankroll_growth import (
    simulate_bankroll_growth,
)


@pytest.fixture
def sample_bets():
    """Creates a sample DataFrame of bets for simulation."""
    data = {
        "tourney_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "odds": [2.5, 1.8, 3.0],
        "winner": [1, 0, 1],  # Win, Loss, Win
        "kelly_fraction": [0.1, 0.05, 0.08],
    }
    return pd.DataFrame(data)


@pytest.fixture
def simulation_params():
    """Provides mock simulation parameters."""
    return {"max_kelly_stake_fraction": 0.1, "max_profit_per_bet": 10000.0}


def test_simulation_flat_strategy(sample_bets, simulation_params):
    """Tests the 'flat' staking strategy."""
    initial_bankroll = 100.0
    stake_unit = 10.0  # Bet 10 units each time

    result_df = simulate_bankroll_growth(
        sample_bets,
        simulation_params,
        initial_bankroll,
        strategy="flat",
        stake_unit=stake_unit,
    )

    # Bet 1 (Win): Stake=10, Profit=10*(2.5-1)=15, Bankroll=100+15=115
    # Bet 2 (Loss): Stake=10, Profit=-10, Bankroll=115-10=105
    # Bet 3 (Win): Stake=10, Profit=10*(3.0-1)=20, Bankroll=105+20=125
    expected_profits = [15.0, -10.0, 20.0]
    expected_bankroll = [115.0, 105.0, 125.0]

    assert result_df["profit"].tolist() == pytest.approx(expected_profits)
    assert result_df["bankroll"].tolist() == pytest.approx(expected_bankroll)
    assert result_df.iloc[-1]["bankroll"] == 125.0


def test_simulation_kelly_strategy(sample_bets, simulation_params):
    """Tests the 'kelly' staking strategy."""
    initial_bankroll = 100.0
    kelly_fraction = 0.5  # Use half-kelly

    result_df = simulate_bankroll_growth(
        sample_bets,
        simulation_params,
        initial_bankroll,
        strategy="kelly",
        kelly_fraction=kelly_fraction,
    )

    # Bet 1 (Win): Kelly=0.1*0.5=0.05, Stake=100*0.05=5, Profit=5*(2.5-1)=7.5, Bankroll=107.5
    # Bet 2 (Loss): Kelly=0.05*0.5=0.025, Stake=107.5*0.025=2.6875, Profit=-2.6875, Bankroll=104.8125
    # Bet 3 (Win): Kelly=0.08*0.5=0.04, Stake=104.8125*0.04=4.1925, Profit=4.1925*(3-1)=8.385, Bankroll=113.1975
    expected_profits = [7.5, -2.6875, 8.385]
    expected_bankroll = [107.5, 104.8125, 113.1975]

    assert result_df["profit"].tolist() == pytest.approx(expected_profits)
    assert result_df["bankroll"].tolist() == pytest.approx(expected_bankroll)
    assert result_df.iloc[-1]["bankroll"] == pytest.approx(113.1975)
