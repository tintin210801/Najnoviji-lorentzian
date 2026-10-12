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
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — koristi High/Low/Close."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Struktura svijeće — koristi Open, High, Low, Close."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    d = pd.DataFrame(index=df.index)
    d["body_pct"]   = (df["Close"] - df["Open"]).abs() / rng
    d["close_pos"]  = (df["Close"] - df["Low"]) / rng
    d["upper_wick"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / rng
    d["lower_wick"] = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / rng
    d["range_pct"]  = rng / df["Close"]
    return d

HORIZON = 3
def volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume kontekst — z-score, ratio, OBV nagib."""
    v = df["Volume"]
    d = pd.DataFrame(index=df.index)
    d["vol_z"]     = (v - v.rolling(period).mean()) / (v.rolling(period).std() + 1e-10)
    d["vol_ratio"] = v / (v.rolling(period).mean() + 1e-10)

    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * v).cumsum()
    d["obv_slope"] = obv.diff(5) / (v.rolling(5).mean() + 1e-10)
    return d
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kreira tehničke indikatore, OHLCV featuree i HMM režime za model."""
    d = df.copy()
    HORIZON = 3

    d["ret_1"] = d["Close"].pct_change(1)
    d["std_10"] = d["Close"].rolling(10).std()

    # HMM režim
    d["hmm_regime"] = apply_rolling_hmm_optimized(d)

    # Target
    d["fwd_return"] = d["Close"].pct_change(HORIZON).shift(-HORIZON)
    d["target"] = np.where(d["fwd_return"].isna(), np.nan, (d["fwd_return"] > 0).astype(float))

    # Klasični indikatori
    d["ret_5"] = d["Close"].pct_change(5)
    d["std_20"] = d["Close"].rolling(20).std()
    d["std_ratio"] = d["std_10"] / (d["std_20"] + 1e-10)

    d["hma_50"] = hma(d["Close"], 50)
    d["price_hma50_ratio"] = d["Close"] / d["hma_50"]

    d["rsi_14"] = rsi(d["Close"], 14)
    d["stoch_rsi"] = stoch_rsi(d["Close"])

    # ▼▼▼ OVI REDCI SU NEDOSTAJALI ▼▼▼
    d["atr_14"] = atr(d, 14)
    d["atr_pct"] = d["atr_14"] / d["Close"]
    d = d.join(candle_features(d))
    d = d.join(volume_features(d))
    # ▲▲▲ KRAJ ▲▲▲

    FEATURES = [
        "stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio",
        "rsi_14", "ret_5", "hmm_regime",
        "atr_pct", "body_pct", "close_pos", "upper_wick", "lower_wick",
        "range_pct", "vol_z", "vol_ratio", "obv_slope",
    ]

    d = d.dropna(subset=FEATURES)
    return d


