# src/tennis_betting_model/utils/alerter.py

import os
import requests
import pandas as pd
from .logger import log_success, log_warning, log_error, log_info


def _send_telegram_message(message: str, parse_mode: str = "Markdown") -> None:
    """Sends a message to the configured Telegram chat."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        log_warning("Telegram credentials not found. Skipping Telegram alert.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        log_success("✅ Alert sent successfully to Telegram.")
    except requests.exceptions.RequestException as e:
        log_warning(f"⚠️ Failed to send Telegram alert: {e}")


def alert_value_bets_found(bet_df: pd.DataFrame) -> None:
    """Formats and sends an alert for newly identified value bets."""
    header = "🚀 ALERT: New Value Bets Found! 🚀"
    message = f"{header}\n\n```\n{bet_df.to_string(index=False)}\n```"
    print(
        "\n"
        + "=" * 50
        + f"\n{header}\n{bet_df.to_string(index=False)}"
        + "\n"
        + "=" * 50
        + "\n"
    )
    _send_telegram_message(message)


def alert_pipeline_success(bets_found: int) -> None:
    """Sends a success message after a pipeline run, only if no bets were found."""
    if bets_found == 0:
        message = "✅ Pipeline run completed successfully. No new value bets found."
        log_info(message)
        # Sparing on alerts, so we don't send a message every 15 mins.
        # This can be enabled if a "heartbeat" is desired.
        # _send_telegram_message(message)


def alert_pipeline_error(error: Exception) -> None:
    """Sends an alert when the pipeline encounters a critical error."""
    header = "❌ CRITICAL: Pipeline Run Failed! ❌"
    message = f"{header}\n\n**Error:**\n`{type(error).__name__}: {error}`"
    log_error(message)
    _send_telegram_message(message)


def alert_bet_placed(order) -> None:
    """Sends a confirmation alert after a bet has been successfully placed."""
    header = "✅ Bet Placed Successfully!"
    instruction = order.instruction_reports[0]
    message = (
        f"{header}\n\n"
        f"**Market ID**: `{order.market_id}`\n"
        f"**Selection ID**: `{instruction.instruction.selection_id}`\n"
        f"**Stake**: `{instruction.instruction.limit_order.size:.2f}`\n"
        f"**Odds**: `{instruction.instruction.limit_order.price}`\n"
        f"**Status**: `{instruction.status}`"
    )
    log_success(message)
    _send_telegram_message(message)
