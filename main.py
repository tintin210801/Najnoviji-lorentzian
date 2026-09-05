"""
main.py (Live / Daily Signals)
-----------------------------
Pojednostavljena skripta za dnevno slanje svježih signala na Telegram
s ugrađenim pravilom minimalnog držanja od 3 dana (3-day hold).
"""

import logging
from datetime import datetime
import pandas as pd
from data_engine import fetch_ohlcv, build_features
from wf_engine import WalkForwardEngine
from models import run_knn_on_fold
from telegram_bot import send_telegram_message

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Lista imovine koju pratimo
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "VELO-USD"]

def main():
    log.info("Započinjem dohvat svježih dnevnih signala za Telegram...")
    
    # Inicijaliziramo Walk-Forward engine
    wf = WalkForwardEngine(train_window=400, test_window=90, step=90)

    # Pripremamo zaglavlje za zajedničku Telegram poruku
    report_lines = [
        "📊 **DNEVNI TRADING SIGNALI** 📊",
        f"📅 *Vrijeme:* `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
    ]

    for symbol in SYMBOLS:
        log.info("=" * 60)
        log.info(f"Obrada simbola: {symbol}")
        
        # 1. Dohvat i obrada podataka (sada uključuje i današnji bar)
        df_raw = fetch_ohlcv(symbol, years_back=4)
        if df_raw.empty:
            log.warning(f"Preskačem {symbol} jer nema podataka.")
            continue
            
        df_feat = build_features(df_raw)
        
        if len(df_feat) < wf.train_window + 30:
            log.warning(f"Nedovoljno podataka za {symbol} (pronađeno {len(df_feat)} redaka). Preskačem.")
            continue

        # 2. Pokretanje Walk-Forward simulacije za dobivanje Out-of-Sample predikcija
        log.info(f"Izvršavam Walk-Forward OOS simulaciju za {symbol}...")
        oos_results = wf.run_walk_forward_simulation(df_feat, run_knn_on_fold)
        
        if oos_results.empty or "position" not in oos_results.columns:
            log.warning(f"Nema OOS rezultata za {symbol}.")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Nema rezultata predikcije*\n")
            continue

        # 3. Primjena pravila minimalnog držanja (3-day hold) i priprema Telegram bloka
        try:
            # Uzimamo zadnje 3 pozicije iz OOS rezultata
            recent_positions = oos_results["position"].tail(3).tolist()
            
            # Pravilo 3-day hold: Ako je unutar zadnja 3 dana bio ulaz (1), drži poziciju
            if 1 in recent_positions:
                current_pos = 1
                status_note = "Aktivno (3-day hold u tijeku)"
            else:
                current_pos = recent_positions[-1] if recent_positions else 0
                status_note = "Izvan pozicije / Čeka signal"

            latest_row = oos_results.iloc[-1]
            last_date = latest_row.name.strftime('%Y-%m-%d') if hasattr(latest_row.name, 'strftime') else str(latest_row.name)[:10]
            conf = latest_row.get("confidence", 0.0)
            close_price = latest_row.get("Close", 0.0)
            if current_pos == 1:
                action = "🟢 **BUY (Long)**"
            else:
                action = "🔴 **SELL (Keš)**"
                
            symbol_block = (
                f"🔹 **{symbol}**\n"
                f"• Akcija: {action}\n"
                f"• Cijena: `{close_price:,.2f}` $\n"
                f"• Konfidencija: `{conf:.2f}`\n"
                f"• Status: {status_note}\n"
                f"• Zadnji bar: `{last_date}`\n"
            )
            report_lines.append(symbol_block)

        except Exception as e:
            log.error(f"Greška pri pripremi Telegram bloka za {symbol}: {e}")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Greška pri obradi signala*\n")

    # 4. Slanje zbirne poruke na Telegram
    final_message = "\n".join(report_lines)
    send_telegram_message(final_message)
    log.info("Dnevni signali uspješno poslani na Telegram!")

if __name__ == "__main__":
    main()
