"""
telegram_bot.py
-----------------------------
Modul zadužen za slanje notifikacija na Telegram kanal/chat.
"""

import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def send_telegram_message(message: str):
    """Šalje tekstualnu poruku na Telegram koristeći environment varijable."""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        log.warning("Telegram token ili chat ID nisu postavljeni. Preskačem slanje poruke.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            log.info("Telegram poruka uspješno poslana.")
        else:
            log.error(f"Greška pri slanju Telegram poruke: {response.text}")
    except Exception as e:
        log.error(f"Iznimka pri slanju Telegram poruke: {e}")
