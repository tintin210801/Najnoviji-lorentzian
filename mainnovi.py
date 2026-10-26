import logging
from datetime import datetime
import pandas as pd

from data_engine import fetch_ohlcv, build_features
from models import run_live_prediction
from telegram_bot import send_telegram_message

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "VELO-USD"]

def main():
    log.info("Započinjem dohvat svježih dnevnih signala za Telegram...")
    
    report_lines = [
        "📊 **DNEVNI TRADING SIGNALI** 📊",
        f"📅 *Vrijeme:* `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
    ]

    for symbol in SYMBOLS:
        log.info("=" * 60)
        log.info(f"Obrada simbola: {symbol}")
        
        # 1. Dohvat i obrada podataka (sada slobodno dohvaća najsvježiji bar)
        df_raw = fetch_ohlcv(symbol, years_back=4)
        if df_raw.empty:
            log.warning(f"Preskačem {symbol} jer nema podataka.")
            continue
            
        df_feat = build_features(df_raw)
        
        if len(df_feat) < 400:
            log.warning(f"Nedovoljno podataka za {symbol}. Preskačem.")
            continue

        # 2. Poziv live Lorentzian predikcije za danas
        try:
            res = run_live_prediction(df_feat)
            if not res:
                report_lines.append(f"🔹 **{symbol}**\n❌ *Greška u live izračunu*\n")
                continue

            current_pos = res["position"]
            conf = res["confidence"]
            close_price = res["close_price"]
            last_date = res["last_date"]
            size = res["size"]

            if current_pos > 0:
                action = "🟢 **BUY (Long)**"
                size_note = f"• Veličina pozicije: `{size*100:.0f}%` kapitala\n"
                status_note = "Aktivna pozicija / Signal je LONG"
            else:
                action = "🔴 **SELL (Keš)**"
                size_note = ""
                status_note = "Izvan pozicije / Čeka signal"

            symbol_block = (
                f"🔹 **{symbol}**\n"
                f"• Akcija: {action}\n"
                f"• Cijena: `{close_price:,.2f}` $\n"
                f"• Konfidencija: `{conf:.2f}`\n"
                f"• Status: {status_note}\n"
                f"{size_note}"
                f"• Zadnji bar: `{last_date}`\n"
            )
            report_lines.append(symbol_block)

        except Exception as e:
            log.error(f"Greška pri obradi {symbol}: {e}")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Greška pri obradi signala*\n")

    # 3. Slanje na Telegram
    final_message = "\n".join(report_lines)
    send_telegram_message(final_message)
    log.info("Dnevni signali uspješno poslani na Telegram!")

if __name__ == "__main__":
    main()
