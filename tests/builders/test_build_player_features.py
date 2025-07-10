import pandas as pd
import pytest
from src.scripts.builders.build_player_features import calculate_player_stats, add_h2h_stats

@pytest.fixture
def sample_match_data() -> pd.DataFrame:
    """Creates a small DataFrame of match results for testing."""
    data = {
        'match_id': [1, 2, 3, 4],
        'tourney_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'surface': ['Hard', 'Hard', 'Clay', 'Hard'],
        'winner_id': [101, 102, 101, 103],
        'loser_id': [102, 103, 103, 101]
    }
    return pd.DataFrame(data)

def test_calculate_player_stats(sample_match_data):
    """
    Tests the player stat calculations for win percentage and form.
    """
    stats_df = calculate_player_stats(sample_match_data)

    # Player 101's stats
    p101_stats = stats_df[stats_df['player_id'] == 101].sort_values('match_id')
    
    # Check win percentages before their third match (match_id 4)
    # Played 2, Won 2 -> 100% win_perc
    assert p101_stats[p101_stats['match_id'] == 4]['win_perc'].iloc[0] == 1.0

    # Player 103's stats
    p103_stats = stats_df[stats_df['player_id'] == 103].sort_values('match_id')

    # Check surface win percentages before their third match (match_id 4)
    # Played 1 on Hard, Won 0 -> 0% surface_win_perc
    assert p103_stats[p103_stats['match_id'] == 4]['surface_win_perc'].iloc[0] == 0.0

    # Check overall win percentages before their third match (match_id 4)
    # Played 2, Won 0 -> 0% win_perc
    assert p103_stats[p103_stats['match_id'] == 4]['win_perc'].iloc[0] == 0.0

@pytest.fixture
def sample_h2h_data() -> pd.DataFrame:
    """Creates a DataFrame to test H2H logic specifically."""
    data = {
        'match_id': ['m1', 'm2', 'm3', 'm4'],
        'tourney_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']),
        'p1_id': [101, 101, 102, 103],
        'p2_id': [102, 102, 103, 101],
        'winner_id': [101, 102, 102, 101] # P1 wins, then P2, P2, P1
    }
    return pd.DataFrame(data)

def test_add_h2h_stats(sample_h2h_data):
    """
    Tests that the H2H statistics are calculated correctly and are point-in-time.
    """
    h2h_df = add_h2h_stats(sample_h2h_data)

    # Match 1 (101 vs 102): First meeting, H2H should be 0-0
    match1 = h2h_df[h2h_df['match_id'] == 'm1'].iloc[0]
    assert match1['h2h_p1_wins'] == 0
    assert match1['h2h_p2_wins'] == 0

    # Match 2 (101 vs 102): Second meeting. P1 (101) won the first match. H2H should be 1-0.
    match2 = h2h_df[h2h_df['match_id'] == 'm2'].iloc[0]
    assert match2['h2h_p1_wins'] == 1
    assert match2['h2h_p2_wins'] == 0

    # Match 4 (103 vs 101): First meeting. H2H should be 0-0.
    match4 = h2h_df[h2h_df['match_id'] == 'm4'].iloc[0]
    assert match4['h2h_p1_wins'] == 0
    assert match4['h2h_p2_wins'] == 0