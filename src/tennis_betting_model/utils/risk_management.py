# src/tennis_betting_model/utils/risk_management.py

from ..utils.logger import log_warning


class RiskManager:
    def __init__(self, max_daily_loss: float, max_exposure: float):
        self.max_daily_loss = max_daily_loss
        self.max_exposure = max_exposure
        self.daily_pnl = 0.0

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def can_place_bet(self, current_exposure: float, new_bet_exposure: float) -> bool:
        if self.daily_pnl <= -self.max_daily_loss:
            log_warning(
                f"Daily loss limit of ${self.max_daily_loss:.2f} reached. Stopping trading."
            )
            return False
        if current_exposure + new_bet_exposure > self.max_exposure:
            log_warning(
                f"Max exposure of ${self.max_exposure:.2f} would be exceeded. Skipping bet."
            )
            return False
        return True
