"""
models.py
-----------------------------
Modul s logikom za Lorentzian KNN model i generiranje predikcija
za Out-of-Sample (OOS) testiranje kroz Walk-Forward engine.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Definiramo iste značajke koje smo koristili u data_engine-u

def lorentzian_distance_vectorized(X_roll: np.ndarray, x_latest: np.ndarray) -> np.ndarray:
    """
    Izračunava Lorentzian udaljenost između povijesnih točaka i trenutne točke.
    Lorentzian udaljenost je otpornija na outliere od Euklidske.
    """
    return np.sum(np.log1p(np.abs(X_roll - x_latest)), axis=1)

FEATURES = [
    "stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio",
    "rsi_14", "ret_5", "hmm_regime",
    "atr_pct", "body_pct", "close_pos", "upper_wick", "lower_wick",
    "range_pct", "vol_z", "vol_ratio", "obv_slope",
]

REGIME_SCALE = {0: 0.3, 1: 0.7, 2: 1.0}  # medvjed / neutral / bik


def run_knn_on_fold(df_train, df_test, K=15, window=300,
                    conf_long=0.60, conf_exit=0.50,
                    target_vol=0.025, max_exposure=1.0) -> pd.DataFrame:
    combined_df = pd.concat([df_train, df_test])
    X_all = combined_df[FEATURES].values
    y_all = combined_df["target"].values

    n_train = len(df_train)
    n_total = len(combined_df)

    sizes = []
    confidences = []

    current_size = 0.0
    hold_counter = 0
    HOLD_BARS = 3

    for i in range(n_train, n_total):
        roll_start = max(0, i - window)
        X_roll = X_all[roll_start:i]
        y_roll = y_all[roll_start:i]
        x_curr = X_all[i]

        if len(X_roll) < K:
            sizes.append(0.0)
            confidences.append(0.0)
            continue

        scaler = StandardScaler()
        X_roll_scaled = scaler.fit_transform(X_roll)
        try:
            x_curr_scaled = scaler.transform(x_curr.reshape(1, -1))[0]
        except Exception:
            sizes.append(0.0)
            confidences.append(0.0)
            continue

        dists = lorentzian_distance_vectorized(X_roll_scaled, x_curr_scaled)
        top_k_idx = np.argsort(dists)[:K]

        # Distance-weighted confidence (bliži susjedi nose više)
        w = 1.0 / (1.0 + dists[top_k_idx])
        w = w / w.sum()
        neighbor_targets = y_roll[top_k_idx]
        valid_mask = ~np.isnan(neighbor_targets)
        if valid_mask.sum() > 0:
            conf = float(np.sum(w[valid_mask] * neighbor_targets[valid_mask]) /
                         np.sum(w[valid_mask]))
        else:
            conf = 0.0
        confidences.append(conf)

        # Odluka o smjeru
        threshold = conf_long if current_size == 0 else conf_exit
        if conf < threshold:
            target_size = 0.0
        else:
            # Confidence scaling — koliko jako vjerujemo
            conf_scalar = np.clip((conf - conf_long) / (0.85 - conf_long), 0.0, 1.0)

            # Vol targeting — koristi atr_pct iz trenutnog retka
            atr_pct = float(combined_df["atr_pct"].iloc[i]) if "atr_pct" in combined_df else 0.02
            vol_scalar = float(np.clip(target_vol / (atr_pct + 1e-10), 0.0, 1.0))

            # Regime scaling
            regime = int(combined_df["hmm_regime"].iloc[i-1]) if i > 0 else 1
            regime_scalar = REGIME_SCALE.get(regime, 0.5)

            target_size = float(np.clip(conf_scalar * vol_scalar * regime_scalar,
                                        0.0, max_exposure))

        # 3-day hold logika na razini veličine
        if hold_counter > 0:
            final_size = current_size
            hold_counter -= 1
        else:
            final_size = target_size
            if (final_size > 0) != (current_size > 0):  # promjena smjera
                hold_counter = HOLD_BARS - 1
            current_size = final_size

        sizes.append(final_size)

    df_result = df_test.copy()
    df_result["confidence"] = confidences
    df_result["size_signal"] = sizes
    # shift(1) — odluka na Close[t], izvršenje na Open[t+1] → u backtestu koristimo position[t] s Open[t+1]
    df_result["position"] = df_result["size_signal"].shift(1).fillna(0.0)
    return df_result

# ==========================================
# TESTING BLOK (Provjera rada s data_engine i wf_engine)
# ==========================================
if __name__ == "__main__":
    from data_engine import fetch_ohlcv, build_features
    from wf_engine import WalkForwardEngine
    
    test_symbol = "BTC-USD"
    df_raw = fetch_ohlcv(test_symbol, years_back=3)
    if not df_raw.empty:
        df_feat = build_features(df_raw)
        
        wf = WalkForwardEngine(train_window=400, test_window=90, step=90)
        
        # Pokrećemo pravu WF simulaciju s našim KNN modelom
        oos_results = wf.run_walk_forward_simulation(df_feat, run_knn_on_fold)
        
        print(f"\nUspješno izračunati OOS rezultati. Ukupno redaka: {len(oos_results)}")
        print(oos_results[["Close", "confidence", "size_signal", "position"]].tail(10))


def run_live_prediction(df_feat: pd.DataFrame, K=15, window=300,
                        conf_long=0.60, conf_exit=0.50,
                        target_vol=0.025, max_exposure=1.0) -> dict:
    """
    Izvršava live inferencu na samom kraju DataFrame-a (najnoviji bar)
    koristeći vašu Lorentzian KNN logiku bez potrebe za Walk-Forward prozorima.
    """
    if len(df_feat) < window:
        log.warning(f"Premalo podataka za live inferencu (pronađeno {len(df_feat)}, traženo {window}).")
        return None

    X_all = df_feat[FEATURES].values
    y_all = df_feat["target"].values
    n_total = len(df_feat)
    
    # Uzimamo zadnji red za predikciju (i = n_total - 1)
    i = n_total - 1
    roll_start = max(0, i - window)
    X_roll = X_all[roll_start:i]
    y_roll = y_all[roll_start:i]
    x_curr = X_all[i]

    if len(X_roll) < K:
        log.warning("Nedovoljno povijesnih susjeda u zadnjem prozoru.")
        return None

    # Skaliranje
    scaler = StandardScaler()
    X_roll_scaled = scaler.fit_transform(X_roll)
    try:
        x_curr_scaled = scaler.transform(x_curr.reshape(1, -1))[0]
    except Exception as e:
        log.error(f"Greška pri skaliranju za live inferencu: {e}")
        return None

    # Izračun Lorentzian udaljenosti i top K susjeda
    dists = lorentzian_distance_vectorized(X_roll_scaled, x_curr_scaled)
    top_k_idx = np.argsort(dists)[:K]

    # Distance-weighted confidence
    w = 1.0 / (1.0 + dists[top_k_idx])
    w = w / w.sum()
    neighbor_targets = y_roll[top_k_idx]
    valid_mask = ~np.isnan(neighbor_targets)
    
    if valid_mask.sum() > 0:
        conf = float(np.sum(w[valid_mask] * neighbor_targets[valid_mask]) / np.sum(w[valid_mask]))
    else:
        conf = 0.0

    # Skaliranje i izračun izloženosti
    conf_scalar = np.clip((conf - conf_long) / (0.85 - conf_long), 0.0, 1.0)
    atr_pct = float(df_feat["atr_pct"].iloc[i]) if "atr_pct" in df_feat else 0.02
    vol_scalar = float(np.clip(target_vol / (atr_pct + 1e-10), 0.0, 1.0))
    regime = int(df_feat["hmm_regime"].iloc[i-1]) if i > 0 else 1
    regime_scalar = REGIME_SCALE.get(regime, 0.5)

    target_size = float(np.clip(conf_scalar * vol_scalar * regime_scalar, 0.0, max_exposure))
    
    # Odluka o poziciji na temelju praga
    if conf >= conf_long and target_size > 0:
        position = 1.0
    else:
        position = 0.0

    last_date = df_feat.index[i].strftime('%Y-%m-%d') if hasattr(df_feat.index[i], 'strftime') else str(df_feat.index[i])[:10]
    
    return {
        "position": position,
        "confidence": conf,
        "size": target_size,
        "last_date": last_date,
        "close_price": float(df_feat["Close"].iloc[i]),
        "regime": regime
    }
