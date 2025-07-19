import pandas as pd
import pytest

from scripts.utils.betting_math import add_ev_and_kelly


def test_add_ev_and_kelly():
    data = {"predicted_prob": [0.5, 0.2, 0.8], "odds": [2.0, 6.0, 1.2]}
    df = pd.DataFrame(data)
    df_processed = add_ev_and_kelly(df, inplace=False)

    # EV = (prob * (odds - 1)) - (1 - prob)
    # Kelly = EV / (odds - 1)

    # Row 1: EV = (0.5 * 1) - 0.5 = 0. Kelly = 0 / 1 = 0.
    assert pytest.approx(df_processed.loc[0, "expected_value"]) == 0.0
    assert pytest.approx(df_processed.loc[0, "kelly_fraction"]) == 0.0

    # Row 2: EV = (0.2 * 5) - 0.8 = 0.2. Kelly = 0.2 / 5 = 0.04
    assert pytest.approx(df_processed.loc[1, "expected_value"]) == 0.2
    assert pytest.approx(df_processed.loc[1, "kelly_fraction"]) == 0.04

    # Row 3: EV = (0.8 * 0.2) - 0.2 = -0.04. Kelly = -0.04 / 0.2 = -0.2
    assert pytest.approx(df_processed.loc[2, "expected_value"]) == -0.04
    assert pytest.approx(df_processed.loc[2, "kelly_fraction"]) == -0.2
