# src/scripts/utils/alerter.py

import os
import requests
from .logger import log_success, log_warning


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> None:
    """
    Sends a formatted message to a specified Telegram chat.

    Args:
        bot_token (str): The token for the Telegram bot.
        chat_id (str): The ID of the Telegram chat to send the message to.
        message (str): The message content to send.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        log_success("✅ Alert sent successfully to Telegram.")
    except requests.exceptions.RequestException as e:
        log_warning(f"⚠️ Failed to send Telegram alert: {e}")


def send_alert(message: str) -> None:
    """
    Sends an alert with the given message to the console and Telegram (if configured).

    Args:
        message (str): The alert message to be sent.
    """
    # --- Console Alert ---
    print("\n" + "=" * 50)
    log_success("🚀 ALERT: New Value Bets Found!")
    print(message)
    print("=" * 50 + "\n")

    # --- Telegram Alert ---
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        send_telegram_message(bot_token, chat_id, message)
    else:
        log_warning("Telegram credentials not found. Skipping Telegram alert.")
