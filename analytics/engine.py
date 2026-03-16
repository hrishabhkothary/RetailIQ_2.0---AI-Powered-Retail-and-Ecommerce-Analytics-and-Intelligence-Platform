# -*- coding: utf-8 -*-
"""
RetailIQ2 — Analytics Engine (Schema-Independent)
====================================================
All analytics functions work from the detected schema, not hardcoded column names.
"""
import pandas as pd
import numpy as np
import logging
from core.data_store import store
from core.schema_detector import parse_dates, coerce_numeric

log = logging.getLogger(__name__)


def _safe(v):
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, pd.Timestamp): return str(v)[:19]
    return v

def _clean_records(records):
    return [{k: _safe(v) for k,v in r.items()} for r in records]

def _df():    return store.get_df()
def _schema(): return store.get_schema()


# ── Overview KPIs ─────────────────────────────────────────────────────────────
def get_kpis():
    df = _df()
    if df.empty: return {}
    sc = _schema()
    amt = sc.get("primary_amount")
    qty = sc.get("primary_qty")
    dt  = sc.get("primary_date")
    cat = sc.get("primary_cat")

    result = {
        "total_rows":    int(len(df)),
        "total_columns": int(len(df.columns)),
        "dataset_name":  store.active or "Unknown",
    }

    if amt:
        s = df[amt].dropna()
        result["total_amount"]   = round(float(s.sum()), 2)
        result["avg_amount"]     = round(float(s.mean()), 2)
        result["max_amount"]     = round(float(s.max()), 2)
        result["min_amount"]     = round(float(s.min()), 2)
        result["amount_label"]   = amt
        # MoM growth
        if dt:
            try:
                d2 = df[[dt, amt]].copy()
                d2[dt] = parse_dates(d2, dt)
                d2["_month"] = d2[dt].dt.to_period("M")
                monthly = d2.groupby("_month")[amt].sum().sort_index()
                if len(monthly) >= 2:
                    curr, prev = float(monthly.iloc[-1]), float(monthly.iloc[-2])
                    result["mom_growth"] = round((curr-prev)/prev*100, 2) if prev else 0
                else:
                    result["mom_growth"] = 0
            except Exception: result["mom_growth"] = 0

    if qty:
        result["total_qty"]   = round(float(df[qty].dropna().sum()), 0)
        result["avg_qty"]     = round(float(df[qty].dropna().mean()), 2)
        result["qty_label"]   = qty

    if cat:
        result["unique_categories"] = int(df[cat].nunique())
        result["cat_label"]         = cat

    if dt:
        try:
            dates = parse_dates(df, dt).dropna()
            if len(dates):
                result["date_from"] = str(dates.min())[:10]
                result["date_to"]   = str(dates.max())[:10]
                result["date_label"] = dt
                result["date_span_days"] = (dates.max() - dates.min()).days
        except Exception: pass

    # Numeric stats for all amount-like cols
    result["numeric_summaries"] = {}
    for c in sc.get("amount_cols",[]) + sc.get("qty_cols",[]) + sc.get("profit_cols",[]):
        s = df[c].dropna()
        if len(s):
            result["numeric_summaries"][c] = {
                "sum":  round(float(s.sum()),2),
                "mean": round(float(s.mean()),2),
                "max":  round(float(s.max()),2),
                "min":  round(float(s.min()),2),
            }

    return result


# ── Time Series ───────────────────────────────────────────────────────────────
def get_time_series(freq="ME"):
    df = _df(); sc = _schema()
    dt  = sc.get("primary_date")
    amt = sc.get("primary_amount")
    if df.empty or not dt: return []
    d2 = df[[dt]].copy()
    d2["_date"] = parse_dates(d2, dt)
    if amt: d2["_val"] = coerce_numeric(df, amt)
    else:   d2["_val"] = 1  # count
    d2 = d2.dropna(subset=["_date"])
    d2 = d2.set_index("_date")
    agg = d2["_val"].resample(freq).sum().reset_index()
    agg.columns = ["period","value"]
    agg["period"] = agg["period"].astype(str)
    return _clean_records(agg.to_dict("records"))


def get_daily_trend(days=90):
    df = _df(); sc = _schema()
    dt  = sc.get("primary_date")
    amt = sc.get("primary_amount")
    if df.empty or not dt: return []
    d2 = df.copy()
    d2["_date"] = parse_dates(d2, dt)
    d2["_val"]  = coerce_numeric(d2, amt) if amt else 1
    d2 = d2.dropna(subset=["_date"])
    cutoff = d2["_date"].max() - pd.Timedelta(days=days)
    d2 = d2[d2["_date"] >= cutoff]
    agg = d2.groupby(d2["_date"].dt.date)["_val"].sum().reset_index()
    agg.columns = ["date","value"]
    agg["date"] = agg["date"].astype(str)
    return _clean_records(agg.to_dict("records"))


# ── Category breakdown ────────────────────────────────────────────────────────
def get_category_breakdown(cat_col=None, val_col=None, top_n=15):
    df = _df(); sc = _schema()
    cat = cat_col or sc.get("primary_cat")
    val = val_col or sc.get("primary_amount")
    if df.empty or not cat: return []
    d2 = df.copy()
    if val:
        d2["_val"] = coerce_numeric(d2, val)
        agg = d2.groupby(cat).agg(
            value   = ("_val","sum"),
            count   = ("_val","count"),
            average = ("_val","mean"),
        ).reset_index()
    else:
        agg = d2.groupby(cat).size().reset_index(name="count")
        agg["value"] = agg["count"]
        agg["average"] = 1
    agg = agg.sort_values("value", ascending=False).head(top_n)
    agg["pct"] = (agg["value"] / agg["value"].sum() * 100).round(1)
    agg.columns = [cat if c==cat else c for c in agg.columns]
    agg = agg.rename(columns={cat: "label"})
    return _clean_records(agg.to_dict("records"))


def get_all_category_cols():
    sc = _schema()
    return sc.get("cat_cols", [])


def get_category_breakdown_by(cat_col, val_col=None, top_n=15):
    return get_category_breakdown(cat_col=cat_col, val_col=val_col, top_n=top_n)


# ── Numeric distributions ─────────────────────────────────────────────────────
def get_numeric_distribution(col=None, bins=20):
    df = _df(); sc = _schema()
    c = col or sc.get("primary_amount")
    if df.empty or not c: return []
    s = coerce_numeric(df, c).dropna()
    if len(s) == 0: return []
    counts, edges = np.histogram(s, bins=bins)
    result = []
    for i in range(len(counts)):
        result.append({
            "bin":   f"{edges[i]:.1f}–{edges[i+1]:.1f}",
            "count": int(counts[i]),
            "from":  round(float(edges[i]),2),
            "to":    round(float(edges[i+1]),2),
        })
    return result


def get_correlation_matrix():
    df = _df(); sc = _schema()
    num_cols = sc.get("amount_cols",[]) + sc.get("qty_cols",[]) + sc.get("profit_cols",[]) + sc.get("numeric_cols",[])
    num_cols = [c for c in num_cols if c in df.columns]
    if len(num_cols) < 2: return {}
    sub = df[num_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(sub) < 5: return {}
    corr = sub.corr().round(3)
    return corr.to_dict()


# ── Top-N analysis ────────────────────────────────────────────────────────────
def get_top_n(group_col=None, val_col=None, n=20, agg="sum"):
    df = _df(); sc = _schema()
    g = group_col or sc.get("primary_cat") or (sc.get("cat_cols") or [None])[0]
    v = val_col   or sc.get("primary_amount")
    if df.empty or not g: return []
    d2 = df.copy()
    if v:
        d2["_val"] = coerce_numeric(d2, v)
        out = d2.groupby(g)["_val"].agg(agg).nlargest(n).reset_index()
        out.columns = ["label","value"]
    else:
        out = d2[g].value_counts().head(n).reset_index()
        out.columns = ["label","value"]
    out["value"] = out["value"].round(2)
    return _clean_records(out.to_dict("records"))


# ── Data profiling ────────────────────────────────────────────────────────────
def get_data_profile():
    df = _df(); sc = _schema()
    if df.empty: return {}
    profile = {
        "n_rows":     sc.get("n_rows",0),
        "n_cols":     sc.get("n_cols",0),
        "columns":    [],
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    for col in df.columns:
        s = df[col]
        null_pct = round(s.isnull().mean()*100,1)
        col_info = {
            "name":     col,
            "role":     sc["col_types"].get(col,"text"),
            "dtype":    sc["col_dtype"].get(col,"object"),
            "nulls":    int(s.isnull().sum()),
            "null_pct": null_pct,
            "unique":   int(s.nunique()),
            "sample":   sc["sample"].get(col,[]),
        }
        if pd.api.types.is_numeric_dtype(s):
            col_info["mean"] = round(float(s.mean()),2) if not s.dropna().empty else None
            col_info["std"]  = round(float(s.std()),2)  if not s.dropna().empty else None
            col_info["min"]  = round(float(s.min()),2)  if not s.dropna().empty else None
            col_info["max"]  = round(float(s.max()),2)  if not s.dropna().empty else None
        profile["columns"].append(col_info)
    return profile


# ── Outlier detection ─────────────────────────────────────────────────────────
def get_outliers(col=None, method="iqr"):
    df = _df(); sc = _schema()
    c = col or sc.get("primary_amount")
    if df.empty or not c: return []
    s = coerce_numeric(df, c).dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)
    else:  # zscore
        z = (s - s.mean()) / s.std()
        mask = z.abs() > 3
    out = df[mask].copy().head(100)
    out["_outlier_value"] = out[c]
    return _clean_records(out[[c]].rename(columns={c:"value"}).head(50).to_dict("records"))


# ── Growth rates ──────────────────────────────────────────────────────────────
def get_growth_table(freq="ME"):
    ts = get_time_series(freq)
    if len(ts) < 2: return []
    result = []
    for i in range(1, len(ts)):
        prev, curr = float(ts[i-1]["value"] or 0), float(ts[i]["value"] or 0)
        growth = round((curr-prev)/prev*100,2) if prev else 0
        result.append({"period": ts[i]["period"], "value": curr, "growth_pct": growth})
    return result


# ── Schema summary ────────────────────────────────────────────────────────────
def get_schema_summary():
    sc = _schema()
    if not sc: return {}
    return {
        "date_cols":    sc.get("date_cols",[]),
        "amount_cols":  sc.get("amount_cols",[]),
        "qty_cols":     sc.get("qty_cols",[]),
        "cat_cols":     sc.get("cat_cols",[]),
        "id_cols":      sc.get("id_cols",[]),
        "numeric_cols": sc.get("numeric_cols",[]),
        "primary_date": sc.get("primary_date"),
        "primary_amount": sc.get("primary_amount"),
        "primary_qty":  sc.get("primary_qty"),
        "primary_cat":  sc.get("primary_cat"),
        "n_rows": sc.get("n_rows",0),
        "n_cols": sc.get("n_cols",0),
    }


# ── Filtered KPIs ─────────────────────────────────────────────────────────────
def get_filtered_kpis(date_from="", date_to="", cat_col="", cat_val=""):
    df = _df().copy()
    sc = _schema()
    if df.empty: return {}
    dt = sc.get("primary_date")
    amt = sc.get("primary_amount")
    # Apply filters
    if dt and date_from:
        df["_d"] = parse_dates(df, dt)
        df = df[df["_d"] >= pd.to_datetime(date_from)]
    if dt and date_to:
        if "_d" not in df.columns: df["_d"] = parse_dates(df, dt)
        df = df[df["_d"] <= pd.to_datetime(date_to)]
    if cat_col and cat_val and cat_col in df.columns:
        df = df[df[cat_col].astype(str) == str(cat_val)]

    result = {"filtered_rows": int(len(df))}
    if amt and len(df):
        s = df[amt].dropna()
        result["total_amount"] = round(float(s.sum()),2)
        result["avg_amount"]   = round(float(s.mean()),2)
        result["amount_label"] = amt
    return result
