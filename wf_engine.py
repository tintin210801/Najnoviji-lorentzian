"""
wf_engine.py
-----------------------------
Modul za Walk-Forward (WF) analizu i rezanje dataseta.
Osigurava strogu odvojenost In-Sample (trening) i Out-of-Sample (test) podataka
te uvodi EMBARGO koji sprječava curenje targeta (HORIZON-dnevni forward return).
"""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Mora odgovarati HORIZON-u u data_engine.py
DEFAULT_EMBARGO = 3


class WalkForwardEngine:
    def __init__(self, train_window: int = 500, test_window: int = 90,
                 step: int = 90, embargo: int = DEFAULT_EMBARGO):
        """
        Args:
            train_window (int): Broj barova za treniranje (In-Sample)
            test_window (int): Broj barova za testiranje (Out-of-Sample)
            step (int): Pomak prozora naprijed u svakoj iteraciji
            embargo (int): Broj barova koji se izbacuju između train i test
                           (mora biti >= HORIZON targeta; inače curenje podataka)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.embargo = embargo

    def split_generator(self, df: pd.DataFrame):
        """
        Generator koji vraća (train, test) DataFrame za svaki WF korak.
        Primjenjuje EMBARGO: zadnjih `embargo` redova treninga se izbacuje
        jer njihov `target` (fwd_return preko HORIZON) zadire u test prozor.
        """
        n = len(df)
        current_idx = self.train_window

        while current_idx < n:
            train_start = current_idx - self.train_window
            train_end = current_idx - self.embargo      # EMBARGO ovdje

            # Preskoči fold ako embargo pojede cijeli trening
            if train_end <= train_start:
                current_idx += self.step
                continue

            df_train = df.iloc[train_start:train_end]

            test_start = current_idx
            test_end = min(current_idx + self.test_window, n)
            df_test = df.iloc[test_start:test_end]

            if len(df_test) == 0:
                break

            yield df_train, df_test
            current_idx += self.step

    def run_walk_forward_simulation(self, df: pd.DataFrame, model_func):
        """
        Izvršava Walk-Forward simulaciju.
        model_func(df_train, df_test) -> DataFrame s OOS rezultatima.
        """
        all_oos_results = []
        fold_id = 1

        for df_train, df_test in self.split_generator(df):
            log.info(f"--- WF Fold {fold_id} ---")
            log.info(f"  IS (Train): {df_train.index[0].date()} do {df_train.index[-1].date()} ({len(df_train)} barova)")
            log.info(f"  OOS (Test): {df_test.index[0].date()} do {df_test.index[-1].date()} ({len(df_test)} barova)")

            fold_result = model_func(df_train, df_test)
            if fold_result is not None and not fold_result.empty:
                all_oos_results.append(fold_result)

            fold_id += 1

        if not all_oos_results:
            log.warning("Nema generiranih WF foldova! Provjerite veličinu dataseta.")
            return pd.DataFrame()

        return pd.concat(all_oos_results)
# ==========================================
# TESTING BLOK (Provjera rada s data_engine)
# ==========================================
if __name__ == "__main__":
    from data_engine import fetch_ohlcv, build_features
    
    test_symbol = "BTC-USD"
    df_raw = fetch_ohlcv(test_symbol, years_back=3)
    if not df_raw.empty:
        df_feat = build_features(df_raw)
        
        wf = WalkForwardEngine(train_window=400, test_window=90, step=90)
        
        # Testna dummy funkcija
        def dummy_model(train, test):
            t = test.copy()
            t["pred_signal"] = 1 
            return t
            
        oos_df = wf.run_walk_forward_simulation(df_feat, dummy_model)
        print(f"\nUspješno spojeno OOS redaka: {len(oos_df)}")
        print(oos_df.tail())
