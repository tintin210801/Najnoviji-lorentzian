"""
data_engine.py
-----------------------------
Modul za dohvat i obradu podataka. 
Osigurava kreiranje značajki (features) strogo bez look-ahead biasa.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler

# Postavke logginga za ovaj modul
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ovdje definiramo globalne konstante vezane isključivo za podatke
HORIZON = 5
HMM_WINDOW = 500
HMM_STEP = 50

# ==========================================
# 1. MATEMATIKA I INDIKATORI
# ==========================================

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

# ==========================================
# 2. HMM REGIMI (STRICTLY OUT-OF-SAMPLE)
# ==========================================

def apply_rolling_hmm(df: pd.DataFrame, window: int = HMM_WINDOW, step: int = HMM_STEP) -> np.ndarray:
    """
    Trenira HMM isključivo na povijesnom prozoru (window) i predviđa 
    režim samo za trenutni dan. Nema curenja iz budućnosti.
    """
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

        # Povremeno ponovno treniranje HMM-a radi performansi
        if i == actual_window or (i - actual_window) % step == 0:
            try:
                model.fit(window_scaled)
                means_ret = model.means_[:, 0]
                state_map = np.argsort(np.argsort(means_ret))
                fitted = True
            except Exception:
                pass # Zadrži prethodni model ako fit pukne

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

# ==========================================
# 3. DOHVAT I SPAJANJE (PIPELINE)
# ==========================================

def fetch_ohlcv(symbol: str, years_back: int = 5) -> pd.DataFrame:
    """Dohvaća dnevne podatke preko yfinance."""
    now_utc = datetime.now(timezone.utc)
    yf_start = (now_utc - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    
    log.info(f"Dohvaćam {symbol} od {yf_start}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=yf_start, interval="1d", auto_adjust=True)

    if df.empty:
        log.error(f"Nema podataka za {symbol}")
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # Brisanje današnjeg (još nezavršenog) bara
    if df.index[-1].date() == now_utc.date():
        df = df.iloc[:-1]
        
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kompajlira sve značajke i target varijablu."""
    d = df.copy()

    # Osnovni returns i std za HMM
    d["ret_1"] = d["Close"].pct_change(1)
    d["std_10"] = d["Close"].rolling(10).std()
    
    # HMM Regimi
    log.info("Računam rolling HMM režime (ovo može potrajati)...")
    d["hmm_regime"] = apply_rolling_hmm(d)

    # Momentum & Volatility Značajke
    d["ret_3"] = d["Close"].pct_change(3)
    d["ret_5"] = d["Close"].pct_change(5)
    d["std_20"] = d["Close"].rolling(20).std()
    
    # Popravljeno dijeljenje s nulom (1e-10)
    d["std_ratio"] = d["std_10"] / (d["std_20"] + 1e-10)

    # HMA
    d["hma_10"] = hma(d["Close"], 10)
    d["hma_20"] = hma(d["Close"], 20)
    d["hma_50"] = hma(d["Close"], 50)
    d["price_hma50_ratio"] = d["Close"] / (d["hma_50"] + 1e-10)
    
    # Volume
    d["volume_ma"] = d["Volume"].rolling(20).mean()
    d["volume_ratio"] = d["Volume"] / (d["volume_ma"] + 1e-10)

    # RSI
    d["rsi_14"] = rsi(d["Close"], 14)
    d["stoch_rsi"] = stoch_rsi(d["Close"])

    # TARGET (Y): Gledamo u budućnost za HORIZON dana. 
    # Ovo se smije koristiti SAMO u In-Sample fazi!
    d["fwd_return"] = d["Close"].pct_change(HORIZON).shift(-HORIZON)
    d["target"] = np.where(d["fwd_return"].isna(), np.nan, (d["fwd_return"] > 0).astype(float))

    # Definiramo konačne stupce koje ćemo koristiti za KNN
    features_list = ["stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio", "rsi_14", "ret_5", "hmm_regime"]
    
    # Mičemo NaN vrijednosti nastale zbog rolling prozora
    d = d.dropna(subset=features_list + ["target"])
    
    log.info(f"Obrada završena. Spreman DataFrame s {len(d)} redaka.")
    return d

# ==========================================
# TESTING BLOK (Pokreće se samo ako direktno runaš ovu skriptu)
# ==========================================
if __name__ == "__main__":
    test_symbol = "BTC-USD"
    df_raw = fetch_ohlcv(test_symbol, years_back=3)
    if not df_raw.empty:
        df_final = build_features(df_raw)
        print("\nPrvih 5 redova (features):")
        print(df_final.head())
        print("\nZadnjih 5 redova (features):")
        print(df_final.tail())