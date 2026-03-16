# -*- coding: utf-8 -*-
"""
RetailIQ2 — In-Memory Data Store
==================================
No MySQL required. All data lives in memory (Pandas DataFrames) + persisted
to Parquet files so data survives server restarts.
Supports multiple datasets uploaded by the user.
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from core.schema_detector import SchemaDetector, parse_dates, coerce_numeric

log = logging.getLogger(__name__)

STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", "store")
os.makedirs(STORE_DIR, exist_ok=True)

_detector = SchemaDetector()


class DataStore:
    """
    Singleton store holding:
      self.datasets  = { name: { 'df': DataFrame, 'schema': dict, 'uploaded_at': str } }
      self.active    = name of active dataset
    """

    def __init__(self):
        self.datasets: dict = {}
        self.active:   str  = None
        self._load_persisted()

    # ── Load / Save ───────────────────────────────────────────────────────────
    def _parquet_path(self, name):
        safe = name.replace("/","_").replace("\\","_").replace(" ","_")
        return os.path.join(STORE_DIR, f"{safe}.parquet")

    def _meta_path(self, name):
        safe = name.replace("/","_").replace("\\","_").replace(" ","_")
        return os.path.join(STORE_DIR, f"{safe}.meta.json")

    def _load_persisted(self):
        for fname in os.listdir(STORE_DIR):
            if not fname.endswith(".parquet"): continue
            name = fname[:-8].replace("_"," ")
            try:
                df   = pd.read_parquet(os.path.join(STORE_DIR, fname))
                meta_p = os.path.join(STORE_DIR, fname.replace(".parquet",".meta.json"))
                meta = {}
                if os.path.exists(meta_p):
                    with open(meta_p) as f: meta = json.load(f)
                schema = _detector.detect(df)
                self.datasets[name] = {"df": df, "schema": schema,
                                       "uploaded_at": meta.get("uploaded_at","unknown"),
                                       "original_name": meta.get("original_name", name)}
                if self.active is None: self.active = name
                log.info("Loaded persisted dataset: %s (%d rows)", name, len(df))
            except Exception as e:
                log.warning("Could not load %s: %s", fname, e)

    def _persist(self, name, df, uploaded_at, original_name):
        try:
            df.to_parquet(self._parquet_path(name), index=False)
            with open(self._meta_path(name), "w") as f:
                json.dump({"uploaded_at": uploaded_at, "original_name": original_name}, f)
        except Exception as e:
            log.warning("Could not persist %s: %s", name, e)

    # ── Public API ────────────────────────────────────────────────────────────
    def ingest(self, df: pd.DataFrame, name: str, original_name: str = "") -> dict:
        """Clean, detect schema, store."""
        df = df.copy()

        # strip whitespace from string columns
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan":"", "None":"", "NaN":""})
            df[c] = df[c].replace("", np.nan)

        schema = _detector.detect(df)

        # parse date columns
        for dc in schema["date_cols"]:
            df[dc] = parse_dates(df, dc)

        # coerce amount / qty / numeric
        for nc in schema["amount_cols"] + schema["qty_cols"] + schema["numeric_cols"] + schema["profit_cols"] + schema["rating_cols"]:
            df[nc] = coerce_numeric(df, nc)

        uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.datasets[name] = {"df": df, "schema": schema,
                               "uploaded_at": uploaded_at,
                               "original_name": original_name or name}
        self.active = name
        self._persist(name, df, uploaded_at, original_name or name)
        log.info("Ingested '%s': %d rows × %d cols", name, len(df), len(df.columns))
        return schema

    def get_df(self, name=None) -> pd.DataFrame:
        n = name or self.active
        if n and n in self.datasets:
            return self.datasets[n]["df"]
        return pd.DataFrame()

    def get_schema(self, name=None) -> dict:
        n = name or self.active
        if n and n in self.datasets:
            return self.datasets[n]["schema"]
        return {}

    def list_datasets(self):
        out = []
        for name, d in self.datasets.items():
            s = d["schema"]
            out.append({"name": name,
                        "original_name": d.get("original_name", name),
                        "rows": s.get("n_rows", 0),
                        "cols": s.get("n_cols", 0),
                        "uploaded_at": d.get("uploaded_at",""),
                        "is_active": name == self.active})
        return out

    def set_active(self, name):
        if name in self.datasets:
            self.active = name
            return True
        return False

    def delete(self, name):
        if name in self.datasets:
            del self.datasets[name]
            try: os.remove(self._parquet_path(name))
            except Exception: pass
            try: os.remove(self._meta_path(name))
            except Exception: pass
            if self.active == name:
                self.active = next(iter(self.datasets), None)
            return True
        return False

    @property
    def has_data(self): return bool(self.datasets) and self.active is not None


# Singleton
store = DataStore()
