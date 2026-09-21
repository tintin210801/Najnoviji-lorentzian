"""
main.py (Live / Daily Signals)
-----------------------------
Skripta za dnevno slanje signala na Telegram s ugrađenim 
pravilom minimalnog držanja od 3 dana (3-day hold) i live predikcijom.
"""

import logging
from datetime import datetime, timezone
import pandas as pd
from data_engine import fetch_ohlcv, build_features
from models import run_knn_on_fold
from telegram_bot import send_telegram_message

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Lista imovine koju pratimo
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "VELO-USD"]

def main():
    log.info("Započinjem dohvat svježih dnevnih signala za Telegram...")
    
    # Pripremamo zaglavlje za zajedničku Telegram poruku
    now_utc = datetime.now(timezone.utc)
    report_lines = [
        "📊 **DNEVNI TRADING SIGNALI** 📊",
        f"📅 *Vrijeme:* `{now_utc.strftime('%Y-%m-%d %H:%M')} UTC`\n"
    ]

    train_window = 400  # Broj dana za treniranje modela

    for symbol in SYMBOLS:
        log.info("=" * 60)
        log.info(f"Obrada simbola: {symbol}")
        
        # 1. Dohvat i obrada podataka (pobrini se da fetch_ohlcv ne briše današnji dan!)
        df_raw = fetch_ohlcv(symbol, years_back=4)
        if df_raw.empty:
            log.warning(f"Preskačem {symbol} jer nema podataka.")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Nema podataka*\n")
            continue
            
        df_feat = build_features(df_raw)
        
        # Treba nam dovoljno podataka za train prozor + zadnjih 3 dana za test
        if len(df_feat) < train_window + 3:
            log.warning(f"Nedovoljno podataka za {symbol} (pronađeno {len(df_feat)} redaka). Preskačem.")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Nedovoljno podataka*\n")
            continue

        # 2. Live predikcija na zadnjem prozoru (Train -> Test zadnja 3 dana)
        try:
            # Dijelimo podatke: train su zadnjih `train_window` dana prije zadnja 3 dana, 
            # a test su zadnja 3 dana (zbog 3-day hold pravila)
            train_df = df_feat.iloc[-(train_window + 3):-3]
            test_df = df_feat.iloc[-3:]
            
            # Pokrećemo model za ovaj zadnji prozor
            oos_results = run_knn_on_fold(train_df, test_df)
            
            if oos_results.empty or "position" not in oos_results.columns:
                log.warning(f"Nema rezultata predikcije za {symbol}.")
                report_lines.append(f"🔹 **{symbol}**\n❌ *Nema rezultata predikcije*\n")
                continue

            # 3. Primjena pravila minimalnog držanja (3-day hold)
            recent_positions = oos_results["position"].tail(3).tolist()
            
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
            
            action = "🟢 **BUY (Long)**" if current_pos == 1 else "🔴 **SELL (Keš)**"
                
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
            log.error(f"Greška pri obradi live signala za {symbol}: {e}")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Greška pri obradi signala*\n")

    # 4. Slanje jedinstvene poruke na Telegram
    final_message = "\n".join(report_lines)
    send_telegram_message(final_message)
    log.info("Dnevni signali uspješno poslani na Telegram!")

if __name__ == "__main__":
    main()
