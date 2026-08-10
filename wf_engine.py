"""
wf_engine.py
-----------------------------
Modul za Walk-Forward (WF) analizu i rezanje dataseta.
Osigurava strogu odvojenost In-Sample (trening) i Out-of-Sample (test) podataka.
"""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class WalkForwardEngine:
    def __init__(self, train_window: int = 500, test_window: int = 90, step: int = 90):
        """
        Args:
            train_window (int): Broj barova za treniranje (In-Sample)
            test_window (int): Broj barova za testiranje (Out-of-Sample)
            step (int): Pomak prozora naprijed u svakoj iteraciji
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def split_generator(self, df: pd.DataFrame):
        """
        Generator koji vraća train i test DataFrame dijelove za svaki WF korak.
        """
        n = len(df)
        current_idx = self.train_window

        while current_idx + self.test_window <= n:
            # In-Sample (Trening) prozor
            train_start = current_idx - self.train_window
            train_end = current_idx
            df_train = df.iloc[train_start:train_end]

            # Out-of-Sample (Test) prozor
            test_start = current_idx
            test_end = current_idx + self.test_window
            df_test = df.iloc[test_start:test_end]

            yield df_train, df_test

            # Pomaknemo se naprijed za definirani step
            current_idx += self.step

    def run_walk_forward_simulation(self, df: pd.DataFrame, model_func):
        """
        Izvršava Walk-Forward simulaciju.
        model_func je funkcija koja prima (df_train, df_test) i vraća OOS rezultate.
        """
        all_oos_results = []
        fold_id = 1

        for df_train, df_test in self.split_generator(df):
            log.info(f"--- WF Fold {fold_id} ---")
            log.info(f"  IS (Train): {df_train.index[0].date()} do {df_train.index[-1].date()} ({len(df_train)} barova)")
            log.info(f"  OOS (Test): {df_test.index[0].date()} do {df_test.index[-1].date()} ({len(df_test)} barova)")

            # Pozivamo model funkciju za treniranje na IS i evaluaciju na OOS
            fold_result = model_func(df_train, df_test)
            if fold_result is not None and not fold_result.empty:
                all_oos_results.append(fold_result)
            
            fold_id += 1

        if not all_oos_results:
            log.warning("Nema generiranih WF foldova! Provjerite veličinu dataseta.")
            return pd.DataFrame()

        # Spajamo sve OOS rezultate u jedan cjeloviti Out-of-Sample DataFrame
        combined_oos = pd.concat(all_oos_results)
        return combined_oos

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