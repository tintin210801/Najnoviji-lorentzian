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

def compute_performance_metrics(df_oos: pd.DataFrame, periods_per_year: int = 365,
                                 cost_bps: float = 10.0) -> dict:
    """Izračunava performanse s realnim Open-to-Open izvršenjem i troškovima."""
    if df_oos.empty or "position" not in df_oos.columns:
        return {}

    df = df_oos.copy()

    # P&L od Open[t] do Open[t+1] — jer ulazimo na otvaranju, ne na zatvaranju
    df["ret_open_next"] = df["Open"].shift(-1) / df["Open"] - 1.0
    df["market_return"] = df["ret_open_next"]
    df["strategy_return"] = df["position"] * df["market_return"]

    # Transakcijski trošak na svaku promjenu pozicije
    turnover = df["position"].diff().abs().fillna(df["position"].abs())
    df["strategy_return"] -= turnover * (cost_bps / 10_000.0)

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

    # Sortino (bolji od Sharpe za asimetrične prinose)
    downside = df.loc[df["strategy_return"] < 0, "strategy_return"].std()
    sortino = (strat_mean / downside) * np.sqrt(periods_per_year) if downside and downside > 0 else 0.0

    # Calmar
    calmar = (total_ret / abs(max_dd)) if max_dd < 0 else 0.0

    # Broj tradeova = broj promjena pozicije
    n_trades = int(turnover.sum() / 2) if turnover.sum() > 0 else 0

    # Win rate po tradeu (ne po baru) — realnije
    trades_pnl = df["strategy_return"][df["position"].diff().fillna(0) != 0]
    # fallback na bar-level ako nema jasnih tradeova
    bar_trades = df[df["strategy_return"] != 0]
    wins = bar_trades[bar_trades["strategy_return"] > 0]
    win_rate = (len(wins) / len(bar_trades) * 100) if len(bar_trades) > 0 else 0.0

    pct_long = (df["position"] > 0).mean() * 100
    avg_exposure = df["position"].mean() * 100

    return {
        "total_return_pct": total_ret,
        "market_return_pct": market_ret,
        "alpha_pct": alpha,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate,
        "pct_long": pct_long,
        "avg_exposure_pct": avg_exposure,
        "n_position_changes": int(turnover.sum()),
        "total_bars": len(df),
    }

import pandas as pd

def print_trade_log(oos_results: pd.DataFrame, symbol: str):
    df = oos_results.copy().reset_index()
    date_col = df.columns[0]

    trades = []
    in_trade = False
    entry_date = entry_price = None

    for i in range(1, len(df)):
        prev_pos = df["position"].iloc[i-1]
        cur_pos  = df["position"].iloc[i]

        if not in_trade and cur_pos > 0 and prev_pos == 0:
            in_trade = True
            entry_date = df[date_col].iloc[i]
            entry_price = df["Open"].iloc[i]           # realan fill
        elif in_trade and cur_pos == 0 and prev_pos > 0:
            in_trade = False
            exit_date = df[date_col].iloc[i]
            exit_price = df["Open"].iloc[i]            # realan fill
            ret_pct = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)[:10],
                'entry_price': entry_price,
                'exit_date': exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date)[:10],
                'exit_price': exit_price,
                'return_pct': ret_pct,
            })

    if in_trade:
        last_date = df[date_col].iloc[-1]
        last_price = df["Close"].iloc[-1]
        ret_pct = ((last_price - entry_price) / entry_price) * 100
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)[:10],
            'entry_price': entry_price,
            'exit_date': '🟢 AKTIVNO (Drži se)',
            'exit_price': last_price,
            'return_pct': ret_pct,
        })

    # ... ispis ostaje isti
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
