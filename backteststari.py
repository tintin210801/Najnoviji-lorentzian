"""
backtest.py
-----------------------------
Modul za računanje financijskih metrika (Sharpe, Max Drawdown, Win Rate)
i evaluaciju Out-of-Sample (OOS) rezultata te osnovne stress testove.
"""

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def compute_performance_metrics(df_oos: pd.DataFrame, periods_per_year: int = 365) -> dict:
    """
    Izračunava ključne performanse strategije na OOS podacima.
    """
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

    # Drawdown izračun
    run_max = df["cum_strategy"].cummax()
    drawdown = (df["cum_strategy"] - run_max) / run_max
    max_dd = drawdown.min() * 100

    # Sharpe ratio
    strat_mean = df["strategy_return"].mean()
    strat_std = df["strategy_return"].std()
    sharpe = (strat_mean / strat_std) * np.sqrt(periods_per_year) if strat_std > 0 else 0.0

    # Win Rate i trgovanja
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

def stress_test_noise(df_oos: pd.DataFrame, noise_level: float = 0.01, simulations: int = 100) -> float:
    """
    Jednostavan Monte Carlo stress test: dodaje Gaussov šum na cijene 
    i provjerava stabilnost Sharpe omjera. Vraća prosječni Sharpe pod šumom.
    """
    sharpes = []
    for _ in range(simulations):
        df_sim = df_oos.copy()
        # Dodajemo nasumičan šum na Close cijene
        noise = np.random.normal(0, noise_level, len(df_sim))
        df_sim["Close_noisy"] = df_sim["Close"] * (1 + noise)
        
        # Računamo prinose s bukom
        strat_ret = df_sim["position"] * df_sim["Close_noisy"].pct_change()
        strat_ret = strat_ret.dropna()
        
        if len(strat_ret) > 0 and strat_ret.std() > 0:
            s = (strat_ret.mean() / strat_ret.std()) * np.sqrt(365)
            sharpes.append(s)
            
    return float(np.mean(sharpes)) if sharpes else 0.0

# ==========================================
# TESTING BLOK (Provjera rada s ostalim modulima)
# ==========================================
if __name__ == "__main__":
    from data_engine import fetch_ohlcv, build_features
    from wf_engine import WalkForwardEngine
    from models import run_knn_on_fold
    
    test_symbol = "BTC-USD"
    df_raw = fetch_ohlcv(test_symbol, years_back=3)
    if not df_raw.empty:
        df_feat = build_features(df_raw)
        wf = WalkForwardEngine(train_window=400, test_window=90, step=90)
        oos_results = wf.run_walk_forward_simulation(df_feat, run_knn_on_fold)
        
        if not oos_results.empty:
            metrics = compute_performance_metrics(oos_results)
            print("\n--- OOS BACKTEST METRIKE ---")
            for k, v in metrics.items():
                print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
                
            st_sharpe = stress_test_noise(oos_results)
            print(f"\nStress Test (Monte Carlo Noise) Prosječni Sharpe: {st_sharpe:.2f}")