"""
backtest.py
-----------------------------
Modul za računanje financijskih metrika, stress testove,
vizualizaciju OOS rezultata i ispis povijesti trgovina (Trade Log).
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def compute_performance_metrics(df_oos: pd.DataFrame, periods_per_year: int = 365) -> dict:
    """Izračunava ključne performanse strategije na OOS podacima."""
    if df_oos.empty or "position" not in df_oos.columns:
        return {}

    df = df_oos.copy()
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["market_return"]
    df = df.dropna(subset=["strategy_return"])

    if len(df) == 0:
        return {}

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["market_return"]).cumprod()

    total_ret = (df["cum_strategy"].iloc[-1] - 1) * 100
    market_ret = (df["cum_market"].iloc[-1] - 1) * 100
    alpha = total_ret - market_ret

    run_max = df["cum_strategy"].cummax()
    drawdown = (df["cum_strategy"] - run_max) / run_max
    max_dd = drawdown.min() * 100

    strat_mean = df["strategy_return"].mean()
    strat_std = df["strategy_return"].std()
    sharpe = (strat_mean / strat_std) * np.sqrt(periods_per_year) if strat_std > 0 else 0.0

    trades = df[df["strategy_return"] != 0]
    wins = trades[trades["strategy_return"] > 0]
    win_rate = (len(wins) / len(trades) * 100) if len(trades) > 0 else 0.0
    pct_long = (df["position"] == 1).mean() * 100

    metrics = {
        "total_return_pct": total_ret,
        "market_return_pct": market_ret,
        "alpha_pct": alpha,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "win_rate_pct": win_rate,
        "pct_long": pct_long,
        "total_bars": len(df)
    }
    return metrics

import pandas as pd

def print_trade_log(oos_results: pd.DataFrame, symbol: str):
    """
    Detaljan ispis svih zatvorenih i aktivnih trgovina na temelju promjene pozicije.
    Prikazuje točan datum ulaza, izlaza, cijene i ostvareni postotak.
    """
    df = oos_results.copy()
    
    # Pridružujemo promjenu pozicije da detektiramo prijelaze 0 -> 1 (Ulaz) i 1 -> 0 (Izlaz)
    df['pos_change'] = df['position'].diff().fillna(0)
    
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    
    for date, row in df.iterrows():
        pos = row['position']
        price = row['Close']
        
        # Ulaz u poziciju (prelazak iz 0 u 1)
        if not in_trade and pos == 1:
            in_trade = True
            entry_date = date
            entry_price = price
            
        # Izlaz iz pozicije (prelazak iz 1 u 0)
        elif in_trade and pos == 0:
            in_trade = False
            exit_date = date
            exit_price = price
            ret_pct = ((exit_price - entry_price) / entry_price) * 100
            
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)[:10],
                'entry_price': entry_price,
                'exit_date': exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date)[:10],
                'exit_price': exit_price,
                'return_pct': ret_pct
            })
            
    # Ako je simulacija završila, a pozicija je i dalje otvorena (još nije došao SELL)
    if in_trade:
        last_date = df.index[-1]
        last_price = df['Close'].iloc[-1]
        ret_pct = ((last_price - entry_price) / entry_price) * 100
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)[:10],
            'entry_price': entry_price,
            'exit_date': '🟢 AKTIVNO (Drži se)',
            'exit_price': last_price,
            'return_pct': ret_pct
        })
        
    # Ispis u konzolu
    print(f"\n" + "=" * 80)
    print(f" 📋 DETALJAN TRADE LOG ZA: {symbol}")
    print("=" * 80)
    
    if not trades:
        print(" Nema zabilježenih trgovina u promatranom razdoblju.")
        print("=" * 80)
        return
        
    print(f"{'DATUM KUPNJE':<15} | {'CIJENA ULAZA':<12} | {'DATUM PRODAJE':<22} | {'CIJENA IZLAZA':<12} | {'PRINOS':<10}")
    print("-" * 80)
    
    for t in trades:
        exit_d = t['exit_date']
        exit_p = f"${t['exit_price']:,.2f}" if isinstance(t['exit_price'], (int, float)) else t['exit_price']
        ent_p = f"${t['entry_price']:,.2f}"
        print(f"{t['entry_date']:<15} | {ent_p:<12} | {exit_d:<22} | {exit_p:<12} | {t['return_pct']:>+.2f}%")
        
    print("=" * 80)

def plot_equity_curve(df_oos: pd.DataFrame, symbol: str):
    """Crtanje usporedbe kumulativnog prinosa strategije i tržišta (OOS)."""
    df = df_oos.copy()
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["market_return"]
    df = df.dropna(subset=["strategy_return"])

    if len(df) == 0:
        return

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["market_return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cum_strategy"], label=f"Lorentzian KNN Strategy ({symbol})", color="blue", linewidth=2)
    plt.plot(df.index, df["cum_market"], label=f"Buy & Hold Market ({symbol})", color="orange", linestyle="--", linewidth=1.5)
    
    plt.title(f"Out-of-Sample (OOS) Walk-Forward Usporedba: {symbol}", fontsize=14, fontweight="bold")
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Kumulativni Multiplikator Kapitala", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

def stress_test_noise(df_oos: pd.DataFrame, noise_level: float = 0.01, simulations: int = 100) -> float:
    """Monte Carlo stress test s gaussovim šumom na cijene."""
    sharpes = []
    for _ in range(simulations):
        df_sim = df_oos.copy()
        noise = np.random.normal(0, noise_level, len(df_sim))
        df_sim["Close_noisy"] = df_sim["Close"] * (1 + noise)
        
        strat_ret = df_sim["position"] * df_sim["Close_noisy"].pct_change()
        strat_ret = strat_ret.dropna()
        
        if len(strat_ret) > 0 and strat_ret.std() > 0:
            s = (strat_ret.mean() / strat_ret.std()) * np.sqrt(365)
            sharpes.append(s)
            
    return float(np.mean(sharpes)) if sharpes else 0.0
