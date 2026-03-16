# -*- coding: utf-8 -*-
"""
RetailIQ2 — Schema Detector
============================
Auto-discovers column roles from ANY uploaded dataset:
  date, amount, quantity, category, id, text, numeric, boolean
No hardcoded column names. Fully schema-independent.
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ── Keyword maps for role detection ──────────────────────────────────────────
DATE_KW    = ["date","time","day","month","year","created","updated","timestamp","dt","ordered","shipped","delivered"]
AMOUNT_KW  = ["amount","price","revenue","sales","total","value","cost","spend","paid","payment","income","profit","earning","gmv","invoice","sum","turnover","fare","fee","charge","rate"]
QTY_KW     = ["qty","quantity","count","units","items","pieces","num","number","volume","sold","ordered","stock"]
CAT_KW     = ["category","cat","type","segment","class","group","department","division","product","item","sku","brand","region","city","state","country","status","gender","channel","source","medium","platform","tier","level","grade"]
ID_KW      = ["id","key","code","ref","reference","order","customer","cust","user","transaction","txn","invoice","serial","no","number"]
RATING_KW  = ["rating","score","stars","review","feedback","satisfaction","nps","grade","rank"]
PROFIT_KW  = ["profit","margin","net","gross","ebit","ebitda","loss"]


def _col_lower(col): return col.lower().replace(" ","_").replace("-","_")

def _matches(col, keywords):
    c = _col_lower(col)
    return any(k in c for k in keywords)

def _looks_like_date(v):
    """Quick check if a string value looks like a date."""
    v = str(v).strip()
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',         # 2024-01-15
        r'\d{2}/\d{2}/\d{4}',          # 01/15/2024
        r'\d{2}-\d{2}-\d{4}',          # 15-01-2024
        r'\d{4}/\d{2}/\d{2}',          # 2024/01/15
        r'\w+ \d{1,2},? \d{4}',        # Jan 15, 2024
        r'\d{1,2} \w+ \d{4}',          # 15 January 2024
        r'\d{4}-\d{2}-\d{2}T',         # ISO 8601
    ]
    for pat in date_patterns:
        if re.search(pat, v):
            return True
    return False


def _is_date_series(series):
    if pd.api.types.is_datetime64_any_dtype(series): return True
    if series.dtype == object:
        sample = series.dropna().head(20).astype(str)
        hits = 0
        for v in sample:
            try:
                pd.to_datetime(v, errors='raise')
                hits += 1
            except Exception: pass
        return hits >= len(sample) * 0.7
    return False

def _is_numeric(series):
    if pd.api.types.is_numeric_dtype(series): return True
    try:
        pd.to_numeric(series.dropna().head(50), errors='raise')
        return True
    except Exception: return False

def _cardinality_ratio(series):
    n = len(series.dropna())
    if n == 0: return 1.0
    return series.nunique() / n


class SchemaDetector:
    """
    Analyses a DataFrame and returns a schema dict:
    {
      'date_cols':    [...],
      'amount_cols':  [...],
      'qty_cols':     [...],
      'cat_cols':     [...],
      'id_cols':      [...],
      'text_cols':    [...],
      'numeric_cols': [...],
      'primary_date': 'colname' or None,
      'primary_amount': 'colname' or None,
      'primary_qty':  'colname' or None,
      'primary_cat':  'colname' or None,
      'all_cols':     [...],
      'n_rows':       int,
      'n_cols':       int,
      'col_types':    {col: role},
      'col_dtype':    {col: dtype_str},
      'sample':       {col: [sample values]},
    }
    """

    def detect(self, df: pd.DataFrame) -> dict:
        schema = {
            "date_cols":    [],
            "amount_cols":  [],
            "qty_cols":     [],
            "cat_cols":     [],
            "id_cols":      [],
            "numeric_cols": [],
            "text_cols":    [],
            "rating_cols":  [],
            "profit_cols":  [],
            "all_cols":     list(df.columns),
            "n_rows":       len(df),
            "n_cols":       len(df.columns),
            "col_types":    {},
            "col_dtype":    {},
            "sample":       {},
        }

        for col in df.columns:
            series  = df[col]
            role    = self._detect_role(col, series)
            dtype   = str(series.dtype)
            schema["col_types"][col] = role
            schema["col_dtype"][col] = dtype
            schema["sample"][col]   = series.dropna().head(3).astype(str).tolist()
            if role == "date":   schema["date_cols"].append(col)
            elif role == "amount": schema["amount_cols"].append(col)
            elif role == "qty":    schema["qty_cols"].append(col)
            elif role == "cat":    schema["cat_cols"].append(col)
            elif role == "id":     schema["id_cols"].append(col)
            elif role == "rating": schema["rating_cols"].append(col)
            elif role == "profit": schema["profit_cols"].append(col)
            elif role == "numeric":schema["numeric_cols"].append(col)
            else:                  schema["text_cols"].append(col)

        # Remove from numeric_cols anything already captured in a specific role
        already = set(
            schema["date_cols"] + schema["amount_cols"] + schema["qty_cols"] +
            schema["cat_cols"]  + schema["id_cols"]     + schema["profit_cols"] +
            schema["rating_cols"]
        )
        schema["numeric_cols"] = [c for c in schema["numeric_cols"] if c not in already]

        # Primary cols (best single representative)
        schema["primary_date"]   = self._pick_primary(schema["date_cols"],   df, "date")
        schema["primary_amount"] = self._pick_primary(schema["amount_cols"],  df, "amount")
        schema["primary_qty"]    = self._pick_primary(schema["qty_cols"],     df, "qty")
        schema["primary_cat"]    = self._pick_primary(schema["cat_cols"],     df, "cat")
        schema["primary_id"]     = self._pick_primary(schema["id_cols"],      df, "id")

        # If no amount found, use biggest numeric col
        if not schema["primary_amount"] and schema["numeric_cols"]:
            best = max(schema["numeric_cols"], key=lambda c: df[c].dropna().mean() if _is_numeric(df[c]) else 0)
            schema["primary_amount"] = best
            schema["amount_cols"] = [best]

        return schema

    def _detect_role(self, col, series):
        # Date check FIRST — structural
        if _is_date_series(series): return "date"
        # Date by name — check before ID
        if _matches(col, DATE_KW):
            if series.dtype == object:
                try:
                    sample = series.dropna().head(10).astype(str)
                    hits = sum(1 for v in sample if _looks_like_date(v))
                    if hits >= len(sample) * 0.6: return "date"
                except Exception: pass
            return "date"  # if name strongly matches date keywords, trust it

        # ID: high cardinality + string/int — but only after ruling out date
        if _matches(col, ID_KW) and _cardinality_ratio(series) > 0.7: return "id"

        # Profit
        if _matches(col, PROFIT_KW) and _is_numeric(series): return "profit"
        # Rating
        if _matches(col, RATING_KW) and _is_numeric(series): return "rating"
        # Amount
        if _matches(col, AMOUNT_KW) and _is_numeric(series): return "amount"
        # Quantity
        if _matches(col, QTY_KW) and _is_numeric(series): return "qty"
        # Category: low cardinality string OR name matches
        if _matches(col, CAT_KW):
            if _cardinality_ratio(series) < 0.5: return "cat"
        if series.dtype == object and _cardinality_ratio(series) < 0.15: return "cat"
        # Generic numeric
        if _is_numeric(series): return "numeric"
        # Low-cardinality string = category (fallback)
        if series.dtype == object and _cardinality_ratio(series) < 0.3: return "cat"
        # Fallback text
        return "text"

    def _pick_primary(self, cols, df, role):
        if not cols: return None
        if len(cols) == 1: return cols[0]
        # Score each candidate
        scores = {}
        for c in cols:
            s = df[c].dropna()
            score = 0
            cl = _col_lower(c)
            # prefer column names that exactly match role keywords
            if role == "date"   and any(k == cl or cl.startswith(k) for k in ["order_date","date","transaction_date","created_at","time"]): score += 10
            if role == "amount" and any(k in cl for k in ["total","amount","revenue","sales","price"]): score += 10
            if role == "qty"    and any(k in cl for k in ["qty","quantity","units","sold"]): score += 10
            if role == "cat"    and any(k in cl for k in ["category","type","segment","product"]): score += 10
            if role == "id"     and any(k in cl for k in ["order","transaction","id"]): score += 5
            # prefer non-null columns
            score += s.count() / len(df) * 5
            scores[c] = score
        return max(scores, key=scores.get)


# ── Convenience: parse & coerce date column ───────────────────────────────────
def parse_dates(df, col):
    """Parse a date column in a DataFrame, returning a datetime Series."""
    try:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.Series([pd.NaT] * len(df), index=df.index)

def coerce_numeric(df, col):
    return pd.to_numeric(df[col], errors='coerce')
