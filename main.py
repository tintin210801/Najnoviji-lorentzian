"""
main.py (Live / Daily Signals)
-----------------------------
Pojednostavljena skripta za dnevno slanje svježih signala na Telegram
korištenjem direktne live inferencije na najnovijem zatvorenom baru.
"""

import logging
from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data_engine import fetch_ohlcv, build_features, HORIZON
from telegram_bot import send_telegram_message

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Lista imovine koju pratimo
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "VELO-USD"]

# Definiramo značajke koje model koristi
FEATURES = [
    "stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio",
    "rsi_14", "ret_5", "hmm_regime",
    "atr_pct", "body_pct", "close_pos", "upper_wick", "lower_wick",
    "range_pct", "vol_z", "vol_ratio", "obv_slope"
]

def main():
    log.info("Započinjem dohvat svježih dnevnih signala za Telegram...")
    
    # Pripremamo zaglavlje za zajedničku Telegram poruku
    report_lines = [
        "📊 **DNEVNI TRADING SIGNALI** 📊",
        f"📅 *Vrijeme:* `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
    ]

    for symbol in SYMBOLS:
        log.info("=" * 60)
        log.info(f"Obrada simbola: {symbol}")
        
        # 1. Dohvat i obrada podataka
        df_raw = fetch_ohlcv(symbol, years_back=4)
        if df_raw.empty:
            log.warning(f"Preskačem {symbol} jer nema podataka.")
            continue
            
        df_feat = build_features(df_raw)
        
        if len(df_feat) < 200:
            log.warning(f"Nedovoljno podataka za {symbol} (pronađeno {len(df_feat)} redaka). Preskačem.")
            continue

        # 2. Live Inference (Direktno treniranje na povijesti i predikcija za najnoviji bar)
        try:
            log.info(f"Izvršavam live inferencu za {symbol}...")
            
            # Odvajamo podatke za trening (maknemo redove gdje nema targeta)
            df_train = df_feat.dropna(subset=["target"]).copy()
            
            X_train = df_train[FEATURES]
            y_train = df_train["target"]
            
            # Uzimamo apsolutno zadnji dostupni red značajki za današnju predikciju
            X_live = df_feat[FEATURES].iloc[[-1]]
            
            # Skaliranje i treniranje KNN modela
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_live_scaled = scaler.transform(X_live)
            
            knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
            knn.fit(X_train_scaled, y_train)
            
            pred_prob = knn.predict_proba(X_live_scaled)[0]
            predicted_class = knn.predict(X_live_scaled)[0]
            confidence = float(pred_prob[int(predicted_class)])
            
            # 3. Priprema Telegram bloka za ovaj simbol
            latest_row = df_feat.iloc[-1]
            last_date = latest_row.name.strftime('%Y-%m-%d') if hasattr(latest_row.name, 'strftime') else str(latest_row.name)[:10]
            close_price = latest_row.get("Close", 0.0)
            
            current_pos = int(predicted_class)
            status_note = "Live Inference (Svježi signal)"

            if current_pos == 1:
                action = "🟢 **BUY (Long)**"
                size_note = "• Veličina pozicije: `100%` kapitala\n"
            else:
                action = "🔴 **SELL (Keš)**"
                size_note = ""

            symbol_block = (
                f"🔹 **{symbol}**\n"
                f"• Akcija: {action}\n"
                f"• Cijena: `{close_price:,.2f}` $\n"
                f"• Konfidencija: `{confidence:.2f}`\n"
                f"• Status: {status_note}\n"
                f"{size_note}"
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
