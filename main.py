"""
main.py
-----------------------------
Glavna skripta koja povezuje sve module:
1. data_engine - Dohvat i obrada podataka
2. wf_engine - Walk-Forward rezanje dataseta
3. models - Lorentzian KNN treniranje i predikcija na OOS
4. backtest - Računanje metrika, stress testovi i vizualizacija
"""

import logging
from datetime import datetime
import pandas as pd
from data_engine import fetch_ohlcv, build_features
from wf_engine import WalkForwardEngine
from models import run_knn_on_fold
from backtest import compute_performance_metrics, stress_test_noise, plot_equity_curve, print_trade_log
from telegram_bot import send_telegram_message

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Lista imovine koju želimo analizirati kroz WF i OOS
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "GOOGL", "ASML", "TSM", "PLTR", "ARM"]

def main():
    log.info("Započinjem lokalnu Walk-Forward (WF) analizu za više imovine...")
    
    # Inicijaliziramo Walk-Forward engine
    wf = WalkForwardEngine(train_window=400, test_window=90, step=90)
    summary_results = []

    # Pripremamo zaglavlje za zajedničku Telegram poruku
    report_lines = [
        "📊 **DNEVNI TRADING IZVJEŠTAJ** 📊",
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
        
        if len(df_feat) < wf.train_window + wf.test_window:
            log.warning(f"Nedovoljno podataka za {symbol} (pronađeno {len(df_feat)} redaka). Preskačem.")
            continue

        # 2. Pokretanje Walk-Forward simulacije (Out-of-Sample predikcije)
        log.info(f"Izvršavam Walk-Forward OOS simulaciju za {symbol}...")
        oos_results = wf.run_walk_forward_simulation(df_feat, run_knn_on_fold)
        
        if oos_results.empty:
            log.warning(f"Nema OOS rezultata za {symbol}.")
            continue

        # 3. Izračun metrika i stress test
        log.info(f"Računam financijske metrike i provodim stress test za {symbol}...")
        metrics = compute_performance_metrics(oos_results)
        noise_sharpe = stress_test_noise(oos_results, noise_level=0.01, simulations=50)
        
        metrics["symbol"] = symbol
        metrics["stress_sharpe"] = noise_sharpe
        summary_results.append(metrics)
        
        # Ispis pojedinačnih rezultata u konzolu
        print(f"\n--- REZULTATI ZA OUT-OF-SAMPLE: {symbol} ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 50)
        
        # Ispis detaljne povijesti trgovina (Trade Log)
        print_trade_log(oos_results, symbol)
        
        # Poziv funkcije za crtanje grafa
        plot_equity_curve(oos_results, symbol)

        # 4. Priprema podataka za Telegram izvještaj (zadnji bar)
        try:
            latest_row = oos_results.iloc[-1]
            last_date = latest_row.name.strftime('%Y-%m-%d') if hasattr(latest_row.name, 'strftime') else str(latest_row.name)[:10]
            current_pos = latest_row["position"]
            conf = latest_row.get("confidence", 0.0)
            
            if current_pos == 1:
                action = "🟢 **BUY (Long)**"
                status_note = "Aktivno (Drži min. 3 dana)"
            else:
                action = "🔴 **SELL (Keš)**"
                status_note = "Izvan pozicije / Čeka signal"
                
            symbol_block = (
                f"🔹 **{symbol}**\n"
                f"• Akcija: {action}\n"
                f"• Konfidencija: `{conf:.2f}`\n"
                f"• Status: {status_note}\n"
                f"• Zadnji bar: `{last_date}`\n"
            )
            report_lines.append(symbol_block)
        except Exception as e:
            log.error(f"Greška pri pripremi Telegram bloka za {symbol}: {e}")
            report_lines.append(f"🔹 **{symbol}**\n❌ *Greška pri dohvaćanju zadnjeg bara*\n")

    # Završni sažetak u konzoli za sve simbole
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        print("\n\n")
        print("=" * 70)
        print(" ZAVRŠNI SAŽETAK SVIH SIMBOLA (OUT-OF-SAMPLE)")
        print("=" * 70)
        print(summary_df[["symbol", "total_return_pct", "market_return_pct", "alpha_pct", "sharpe_ratio", "max_drawdown_pct", "stress_sharpe"]].to_string(index=False))
        print("=" * 70)

        # Slanje jedinstvene, zbirne poruke na Telegram
        final_message = "\n".join(report_lines)
        send_telegram_message(final_message)

if __name__ == "__main__":
    main()
