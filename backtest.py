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

def print_trade_log(df_oos: pd.DataFrame, symbol: str):
    """Pronazi i ispisuje svaku promjenu pozicije (BUY / SELL) s cijenama i datumima."""
    df = df_oos.copy()
    # Detektiramo promjenu pozicije (1 -> 0 ili 0 -> 1)
    df["trade_action"] = df["position"].diff()
    
    # Filtriramo samo retke gdje se akcija dogodila
    trades = df[df["trade_action"].isin([1, -1])].copy()
    
    if trades.empty:
        log.info(f"Nema zabilježenih trgovina za {symbol}.")
        return

    print(f"\n" + "=" * 65)
    print(f" POVIJEST TRGOVINA (TRADE LOG) ZA: {symbol}")
    print("=" * 65)
    
    trade_list = []
    for idx, row in trades.iterrows():
        action = "BUY (Ulaz u Long)" if row["trade_action"] == 1 else "SELL (Izlaz iz Longa)"
        price = row["Close"]
        conf = row.get("confidence", 0.0)
        
        # Formatiramo datum
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        
        trade_list.append({
            "Datum": date_str,
            "Akcija": action,
            "Cijena ($)": f"{price:,.2f}",
            "Konfidencija": f"{conf:.2f}"
        })
        
    df_trades = pd.DataFrame(trade_list)
    print(df_trades.to_string(index=False))
    print("=" * 65 + "\n")

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