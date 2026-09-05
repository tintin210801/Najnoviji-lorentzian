"""
data_engine.py
-----------------------------
Modul za dohvat OHLCV podataka i inženjering značajki (Feature Engineering)
uključujući HMM (Hidden Markov Models) filtre režima tržišta.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def fetch_ohlcv(symbol: str, years_back: int = 4) -> pd.DataFrame:
    """Dohvaća dnevne OHLCV podatke preko yfinance."""
    now_utc = datetime.now(timezone.utc)
    yf_start = (now_utc - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    yf_end = (now_utc + timedelta(days=2)).strftime("%Y-%m-%d")

    log.info("Dohvaćam podatke za %s (od %s do %s)...", symbol, yf_start, yf_end)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=yf_start, end=yf_end, interval="1d", auto_adjust=True)

    if df.empty:
        log.warning(f"yfinance vratio prazan DataFrame za {symbol}")
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Ukloni današnji nezavršeni bar ako je trgovački dan u tijeku
    today_utc = now_utc.date()
    if not df.empty and df.index[-1].date() == today_utc:
        df = df.iloc[:-1]
    print(f"\n--- ZADNJI REDCI ZA {symbol} (YFINANCE) ---")
    print(df.tail(10))
    return df

def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    
def hma(series: pd.Series, period: int) -> pd.Series:
    half = int(period / 2)
    sqrt_ = int(np.sqrt(period))
    return wma(2 * wma(series, half) - wma(series, period), sqrt_)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))

def stoch_rsi(series: pd.Series, period: int = 14, smooth_k: int = 3) -> pd.Series:
    r = rsi(series, period)
    r_min = r.rolling(period).min()
    r_max = r.rolling(period).max()
    stoch = (r - r_min) / (r_max - r_min + 1e-10)
    return stoch.rolling(smooth_k).mean()

def apply_rolling_hmm_optimized(df: pd.DataFrame, window=500, step=50) -> np.ndarray:
    """Racuna HMM režime tržišta kroz klizeći prozor bez curenja podataka."""
    data = df[["ret_1", "std_10"]].fillna(0).values
    n = len(data)
    regimes = np.zeros(n, dtype=int)

    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=50, random_state=42)
    state_map = [0, 1, 2]
    fitted = False
    actual_window = min(window, n - 1)

    for i in range(actual_window, n):
        start_idx = max(0, i - actual_window)
        window_data = data[start_idx:i]

        scaler = StandardScaler()
        window_scaled = scaler.fit_transform(window_data)

        if i == actual_window or (i - actual_window) % step == 0:
            try:
                model.fit(window_scaled)
                means_ret = model.means_[:, 0]
                state_map = np.argsort(np.argsort(means_ret)) # 0 = medvjed, 2 = bik
                fitted = True
            except Exception as e:
                log.warning(f"HMM fit failed at i={i}: {e}")

        if fitted:
            current_point = data[i].reshape(1, -1)
            current_scaled = scaler.transform(current_point)
            try:
                raw_pred = model.predict(current_scaled)[0]
                regimes[i] = state_map[raw_pred]
            except Exception:
                regimes[i] = 0
        else:
            regimes[i] = 0

    return regimes

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kreira tehničke indikatore i HMM režime za model."""
    d = df.copy()
    HORIZON = 5

    d["ret_1"] = d["Close"].pct_change(1)
    d["std_10"] = d["Close"].rolling(10).std()
    
    # HMM Režim kao važna značajka
    d["hmm_regime"] = apply_rolling_hmm_optimized(d)

    # Target
    d["fwd_return"] = d["Close"].pct_change(HORIZON).shift(-HORIZON)
    d["target"] = np.where(d["fwd_return"].isna(), np.nan, (d["fwd_return"] > 0).astype(float))

    d["ret_5"] = d["Close"].pct_change(5)
    d["std_20"] = d["Close"].rolling(20).std()
    d["std_ratio"] = d["std_10"] / (d["std_20"] + 1e-10)

    d["hma_50"] = hma(d["Close"], 50)
    d["price_hma50_ratio"] = d["Close"] / d["hma_50"]

    d["rsi_14"] = rsi(d["Close"], 14)
    d["stoch_rsi"] = stoch_rsi(d["Close"])

    
    # Lista značajki koje model koristi
    FEATURES = ["stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio", "rsi_14", "ret_5", "hmm_regime"]
    
    # IZMJENA OVDJE: 
    # Brišemo samo retke gdje fale osnovne značajke (FEATURES), 
    # ali dopuštamo da 'target' bude NaN na samom kraju (za današnji/najnoviji bar)
    d = d.dropna(subset=FEATURES)
    
    return d


