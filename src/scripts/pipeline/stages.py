# src/scripts/pipeline/stages.py

"""
Defines the structure and components of the data processing pipeline.
"""

from scripts.builders.core import build_matches_from_snapshots
from scripts.builders.build_player_features import build_player_features
from scripts.pipeline.build_odds_features import build_odds_features
from scripts.pipeline.detect_value_bets import detect_value_bets
from scripts.pipeline.match_selection_ids import assign_selection_ids
from scripts.pipeline.merge_final_ltps_into_matches import merge_final_ltps
from scripts.pipeline.predict_win_probs import predict_win_probs
from scripts.pipeline.simulate_bankroll_growth import simulate_bankroll_growth

# Defines the sequence of functions, their inputs, and outputs for the pipeline.
# This structure is imported by the pipeline orchestrator.
STAGE_FUNCS = {
    "build": {
        "fn": build_matches_from_snapshots,
        "input_keys": ["snapshots_csv"],
        "output_key": "matches_csv",
    },
    "ids": {
        "fn": assign_selection_ids,
        "input_keys": ["matches_csv", "sackmann_csv", "snapshots_csv"],
        "output_key": "matches_with_ids_csv",
    },
    "player_features": {
        "fn": build_player_features,
        "input_keys": ["sackmann_csv", "snapshots_csv"],  # MODIFIED: Added snapshots_csv
        "output_key": "player_features_csv",
    },
    "merge": {
        "fn": merge_final_ltps,
        "input_keys": ["matches_with_ids_csv", "snapshots_csv"],
        "output_key": "merged_matches_csv",
    },
    "features": {
        "fn": build_odds_features,
        "input_keys": ["merged_matches_csv", "player_features_csv"],
        "output_key": "features_csv",
    },
    "predict": {
        "fn": predict_win_probs,
        "input_keys": ["model_file", "features_csv"],
        "output_key": "predictions_csv",
    },
    "detect": {
        "fn": detect_value_bets,
        "input_keys": ["predictions_csv"],
        "output_key": "value_bets_csv",
    },
    "simulate": {
        "fn": simulate_bankroll_growth,
        "input_keys": ["value_bets_csv"],
        "output_key": "simulation_csv",
    },
}