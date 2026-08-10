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
FEATURES = ["stoch_rsi", "std_20", "price_hma50_ratio", "std_ratio", "rsi_14", "ret_5", "hmm_regime"]

def lorentzian_distance_vectorized(X_roll: np.ndarray, x_latest: np.ndarray) -> np.ndarray:
    """
    Izračunava Lorentzian udaljenost između povijesnih točaka i trenutne točke.
    Lorentzian udaljenost je otpornija na outliere od Euklidske.
    """
    return np.sum(np.log1p(np.abs(X_roll - x_latest)), axis=1)

def run_knn_on_fold(df_train: pd.DataFrame, df_test: pd.DataFrame, K: int = 15, window: int = 300, conf_long: float = 0.65, conf_exit: float = 0.55) -> pd.DataFrame:
    """
    Funkcija koju WalkForwardEngine poziva za svaki fold.
    Za svaki dan u testnom (OOS) skupu, model trenira (skalira) na zadnjih 'window' dana 
    (uzimajući u obzir i kraj trening skupa) i daje predikciju za taj dan.
    """
    # Spajamo kraj treninga i cijeli test kako bismo imali dovoljno povijesti za rolling prozore u testu
    combined_df = pd.concat([df_train, df_test])
    
    X_all = combined_df[FEATURES].values
    y_all = combined_df["target"].values
    
    n_train = len(df_train)
    n_total = len(combined_df)
    
    preds = []
    confidences = []
    
    current_pos = 0
    hold_counter = 0
    HOLD_BARS = 3  # Koliko minimalno barova držimo poziciju

    # Iteriramo kroz svaki dan u Out-of-Sample (testnom) dijelu
    for i in range(n_train, n_total):
        # Definišemo rolling prozor za učenje (unazad 'window' dana od trenutnog dana i)
        roll_start = max(0, i - window)
        
        X_roll = X_all[roll_start:i]
        y_roll = y_all[roll_start:i]
        x_curr = X_all[i]
        
        # Sigurnosna provjera da imamo dovoljno podataka u prozoru
        if len(X_roll) < K:
            preds.append(0)
            confidences.append(0.0)
            continue

        # Skaliranje isključivo na trenutnom rolling prozoru (sprečavanje curenja podataka)
        scaler = StandardScaler()
        X_roll_scaled = scaler.fit_transform(X_roll)
        
        try:
            x_curr_scaled = scaler.transform(x_curr.reshape(1, -1))[0]
        except Exception:
            preds.append(0)
            confidences.append(0.0)
            continue

        # Izračun udaljenosti i traženje K susjeda
        dists = lorentzian_distance_vectorized(X_roll_scaled, x_curr_scaled)
        top_k_idx = np.argsort(dists)[:K]
        
        # Konfidencija je prosjek ciljnih vrijednosti K najbližih susjeda
        conf = float(np.mean(y_roll[top_k_idx]))
        confidences.append(conf)

        # Generiranje sirovog signala na temelju konfidencije i praga
        raw_signal = 1 if conf >= (conf_long if current_pos == 0 else conf_exit) else 0

        # Upravljanje trajanjem pozicije (Hold Bars)
        if hold_counter > 0:
            final_signal = current_pos
            hold_counter -= 1
        else:
            final_signal = raw_signal
            if final_signal != current_pos:
                hold_counter = HOLD_BARS - 1
                current_pos = final_signal

        preds.append(final_signal)

    # Vraćamo testni DataFrame obogaćen predikcijama
    df_result = df_test.copy()
    df_result["pred"] = preds
    df_result["confidence"] = confidences
    
    # Pozicija se pomiče za 1 bar unaprijed (trgujemo dan nakon generiranog signala)
    df_result["position"] = df_result["pred"].shift(1).fillna(0)
    
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
        print(oos_results[["Close", "pred", "confidence", "position"]].tail(10))