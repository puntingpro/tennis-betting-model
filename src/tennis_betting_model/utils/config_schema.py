# src/tennis_betting_model/utils/config_schema.py

from pydantic import BaseModel
from typing import List


class DataPaths(BaseModel):
    raw_data_dir: str
    processed_data_dir: str
    plot_dir: str
    raw_players: str
    consolidated_rankings: str
    betfair_raw_odds: str
    player_map: str
    betfair_match_log: str
    elo_ratings: str
    consolidated_features: str
    backtest_market_data: str
    model: str
    backtest_results: str
    tournament_summary: str


class EloConfig(BaseModel):
    k_factor: int
    rating_diff_factor: int


class MappingParams(BaseModel):
    confidence_threshold: int


class TrainingParams(BaseModel):  # Renamed from ModelParams
    hyperparameter_trials: int
    max_training_samples: int | None = None


class Betting(BaseModel):
    ev_threshold: float
    confidence_threshold: float
    betfair_commission: float
    live_bankroll: float
    live_kelly_fraction: float
    max_kelly_stake_fraction: float
    profitable_tournaments: List[str]


class AnalysisStrategy(BaseModel):
    name: str
    min_odds: float = 0.0
    max_odds: float = 1000.0
    min_ev: float = 0.0


class AnalysisParams(BaseModel):
    min_bets_for_summary: int
    leaderboard_top_n: int


class SimulationParams(BaseModel):
    max_kelly_stake_fraction: float
    max_profit_per_bet: float


class Config(BaseModel):
    data_paths: DataPaths
    elo_config: EloConfig
    mapping_params: MappingParams
    training_params: TrainingParams  # Renamed from model_params
    betting: Betting
    analysis_strategies: List[AnalysisStrategy]
    analysis_params: AnalysisParams
    simulation_params: SimulationParams
