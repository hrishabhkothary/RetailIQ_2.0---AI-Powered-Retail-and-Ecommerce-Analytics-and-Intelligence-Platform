# -*- coding: utf-8 -*-
"""
RetailIQ2 — ML Pipeline (Schema-Independent)
==============================================
All models discover their input features from the schema detector.
No hardcoded column names anywhere.
"""
import os, pickle, logging, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from core.data_store import store
from core.schema_detector import parse_dates, coerce_numeric

log = logging.getLogger(__name__)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(obj, name):
    p = os.path.join(MODEL_DIR, name+".pkl")
    with open(p,"wb") as f: pickle.dump(obj,f,protocol=4)
    log.info("Saved model: %s (%.1f KB)", name, os.path.getsize(p)/1024)

def load_model(name):
    p = os.path.join(MODEL_DIR, name+".pkl")
    if not os.path.exists(p): return None
    with open(p,"rb") as f: return pickle.load(f)

_metrics_log = []   # in-memory metrics history

def log_metric(model, metrics: dict):
    from datetime import datetime
    _metrics_log.append({"model": model, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics})

def get_all_metrics():
    return list(reversed(_metrics_log))


# ── 1. Demand / Value Forecasting ─────────────────────────────────────────────
def train_forecast():
    """
    Trains a RandomForest to predict the primary amount column
    using time features derived from the primary date column.
    Works on ANY dataset with a date and numeric column.
    """
    df = store.get_df(); sc = store.get_schema()
    dt  = sc.get("primary_date")
    amt = sc.get("primary_amount")
    if df.empty or not dt or not amt:
        log.warning("Forecast: need a date column and a numeric column.")
        return {"error": "Need at least one date column and one numeric column."}

    d = df.copy()
    d["_date"] = parse_dates(d, dt)
    d["_val"]  = coerce_numeric(d, amt)
    d = d.dropna(subset=["_date","_val"])
    if len(d) < 30:
        return {"error": "Need at least 30 rows with valid date and numeric values."}

    # Aggregate — try daily first, fall back to weekly/monthly if too few points
    daily = d.groupby(d["_date"].dt.date)["_val"].sum().reset_index()
    daily.columns = ["date","value"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Determine aggregation granularity
    n_daily = len(daily)
    if n_daily >= 30:
        d = daily
        forecast_freq = "D"
        lag_short, lag_long, roll_win = 1, 7, 7
    elif n_daily >= 10:
        # Aggregate to weekly
        d = daily.set_index("date")["value"].resample("W").sum().reset_index()
        d.columns = ["date","value"]
        forecast_freq = "W"
        lag_short, lag_long, roll_win = 1, 4, 4
    else:
        # Aggregate to monthly
        d = daily.set_index("date")["value"].resample("ME").sum().reset_index()
        d.columns = ["date","value"]
        forecast_freq = "M"
        lag_short, lag_long, roll_win = 1, 3, 3

    if len(d) < 8:
        return {"error": f"Not enough time points ({len(d)}) for forecasting. Upload more data."}

    # Feature engineering
    d["dow"]    = d["date"].dt.dayofweek
    d["month"]  = d["date"].dt.month
    d["dom"]    = d["date"].dt.day
    d["week"]   = d["date"].dt.isocalendar().week.astype(int)
    d["quarter"]= d["date"].dt.quarter
    d[f"lag{lag_short}"]  = d["value"].shift(lag_short)
    d[f"lag{lag_long}"]   = d["value"].shift(lag_long)
    d[f"roll{roll_win}"]  = d["value"].rolling(roll_win).mean()
    d = d.dropna()

    feats = ["dow","month","dom","week","quarter",
             f"lag{lag_short}",f"lag{lag_long}",f"roll{roll_win}"]
    X = d[feats]; y = d["value"]
    if len(X) < 6: return {"error": "Not enough data after feature engineering."}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae  = round(float(mean_absolute_error(y_te, y_pred)),4)
    rmse = round(float(np.sqrt(mean_squared_error(y_te, y_pred))),4)
    r2   = round(float(r2_score(y_te, y_pred)),4)

    # Generate 30-period forecast
    last_date = d["date"].max()
    roll_buf  = list(d["value"].values[-max(roll_win, lag_long):])
    forecasts = []
    freq_delta = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}
    delta = freq_delta.get(forecast_freq, pd.Timedelta(days=1))

    for i in range(1, 31):
        fd = last_date + (delta * i) if forecast_freq != "M" else last_date + pd.DateOffset(months=i)
        lag_s_val  = roll_buf[-lag_short] if len(roll_buf) >= lag_short else roll_buf[0]
        lag_l_val  = roll_buf[-lag_long]  if len(roll_buf) >= lag_long  else roll_buf[0]
        roll_val   = float(np.mean(roll_buf[-roll_win:])) if len(roll_buf) >= roll_win else float(np.mean(roll_buf))
        row = pd.DataFrame([[fd.dayofweek, fd.month, fd.day,
                              fd.isocalendar()[1], (fd.month-1)//3+1,
                              lag_s_val, lag_l_val, roll_val]], columns=feats)
        pred = float(max(0, model.predict(row)[0]))
        forecasts.append({
            "date":      str(fd.date()),
            "predicted": round(pred, 2),
            "lower":     round(pred * 0.82, 2),
            "upper":     round(pred * 1.18, 2),
        })
        roll_buf.append(pred)

    save_model({"model":model,"feats":feats,"last_date":str(last_date.date())}, "forecast_rf")
    metrics = {"MAE":mae,"RMSE":rmse,"R2":r2,"col_used":amt,"date_used":dt}
    log_metric("RandomForest_Forecast", metrics)
    log.info("Forecast trained — MAE=%.4f R2=%.4f", mae, r2)
    return {"mae":mae,"rmse":rmse,"r2":r2,"forecasts":forecasts,
            "amount_col":amt,"date_col":dt}


def get_forecasts():
    m = load_model("forecast_rf")
    if not m: return []
    # Return stored forecasts by re-running on loaded model
    r = train_forecast()
    if "forecasts" in r: return r["forecasts"]
    return []


# ── 2. Clustering / Segmentation ──────────────────────────────────────────────
def train_segmentation(n_clusters=5):
    """
    Clusters rows using all available numeric columns.
    Assigns a segment label to each row.
    """
    df = store.get_df(); sc = store.get_schema()
    seen=set()
    num_cols=[]
    for c in (sc.get("amount_cols",[])+sc.get("qty_cols",[])+sc.get("numeric_cols",[])+sc.get("profit_cols",[])):
        if c not in seen and c in df.columns: seen.add(c); num_cols.append(c)

    if df.empty or len(num_cols) < 2:
        return {"error": "Need at least 2 numeric columns for segmentation."}

    d = df[num_cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna()
    if len(d) < n_clusters*3:
        return {"error": f"Not enough valid rows for {n_clusters} clusters."}

    scaler = StandardScaler()
    X = scaler.fit_transform(d)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    d["_cluster"] = labels

    # Rank clusters by mean of the first amount/numeric column
    rank_col = num_cols[0]
    cluster_means = d.groupby("_cluster")[rank_col].mean().sort_values(ascending=False)
    seg_names = ["High-Value","Above-Average","Average","Below-Average","Low-Value",
                 "Segment-6","Segment-7","Segment-8","Segment-9","Segment-10"]
    label_map = {int(c): seg_names[i] if i < len(seg_names) else f"Segment-{i}"
                 for i, c in enumerate(cluster_means.index)}

    d["_segment"] = d["_cluster"].map(label_map)
    seg_summary = d.groupby("_segment")[num_cols].agg(["mean","count"]).round(2)

    # Flat result for API
    result_segs = []
    for seg in d["_segment"].unique():
        sub = d[d["_segment"]==seg]
        row = {"segment": seg, "count": int(len(sub))}
        for c in num_cols:
            row[f"avg_{c}"] = round(float(sub[c].mean()),2)
        result_segs.append(row)
    result_segs.sort(key=lambda x: x.get(f"avg_{num_cols[0]}",0), reverse=True)

    save_model({"km":km,"scaler":scaler,"label_map":label_map,"num_cols":num_cols}, "segmentation_km")
    metrics = {"inertia":round(float(km.inertia_),2),"clusters":n_clusters,"cols_used":len(num_cols)}
    log_metric("KMeans_Segmentation", metrics)
    log.info("Segmentation trained — %d clusters, %d rows", n_clusters, len(d))
    return {"segments":result_segs,"num_cols":num_cols,"n_clusters":n_clusters}


def get_segments():
    r = train_segmentation()
    return r.get("segments",[]) if "segments" in r else []


# ── 3. Anomaly Detection ──────────────────────────────────────────────────────
def train_anomaly_detection(contamination=0.05):
    """
    Isolation Forest on all numeric columns.
    Flags statistical outliers — works on any numeric dataset.
    """
    df = store.get_df(); sc = store.get_schema()
    seen=set()
    num_cols=[]
    for c in (sc.get("amount_cols",[])+sc.get("qty_cols",[])+sc.get("numeric_cols",[])+sc.get("profit_cols",[])):
        if c not in seen and c in df.columns: seen.add(c); num_cols.append(c)

    if df.empty or len(num_cols) == 0:
        return {"error": "Need at least one numeric column for anomaly detection."}

    d = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 20: return {"error": "Need at least 20 rows for anomaly detection."}

    scaler = StandardScaler()
    X = scaler.fit_transform(d)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    d["_anomaly"]       = iso.fit_predict(X)
    d["_anomaly_score"] = iso.score_samples(X)

    anomalies = d[d["_anomaly"]==-1].copy()
    log.info("Anomaly: flagged %d/%d (%.1f%%)", len(anomalies), len(d), len(anomalies)/len(d)*100)

    result = []
    for _, row in anomalies.head(200).iterrows():
        r = {"anomaly_score": round(float(row["_anomaly_score"]),4)}
        for c in num_cols: r[c] = round(float(row[c]),2)
        result.append(r)

    result.sort(key=lambda x: x["anomaly_score"])

    save_model({"iso":iso,"scaler":scaler,"num_cols":num_cols}, "anomaly_iso")
    metrics = {"flagged":len(anomalies),"total":len(d),"pct":round(len(anomalies)/len(d)*100,2)}
    log_metric("IsolationForest_Anomaly", metrics)
    return {"anomalies":result,"total_scored":len(d),"flagged":len(anomalies),"num_cols":num_cols}


def get_anomalies():
    r = train_anomaly_detection()
    return r.get("anomalies",[]) if "anomalies" in r else []


# ── 4. Trend Analysis ─────────────────────────────────────────────────────────
def run_trend_analysis():
    """In-memory rolling stats — no .pkl saved."""
    df = store.get_df(); sc = store.get_schema()
    dt  = sc.get("primary_date")
    amt = sc.get("primary_amount")
    if df.empty or not dt: return {"error":"Need a date column."}

    d = df.copy()
    d["_date"] = parse_dates(d, dt)
    d["_val"]  = coerce_numeric(d, amt) if amt else pd.Series(np.ones(len(d)))
    d = d.dropna(subset=["_date"])
    d = d.set_index("_date")["_val"].resample("ME").sum()

    if len(d) < 2: return {"error":"Not enough data for trend analysis."}

    pct_change   = d.pct_change()*100
    rolling3     = d.rolling(3).mean()
    latest_growth= round(float(pct_change.dropna().iloc[-1]),2) if len(pct_change.dropna()) else 0
    avg_monthly  = round(float(d.mean()),2)
    volatility   = round(float(d.std()/d.mean()*100),2) if d.mean() else 0
    peak_period  = str(d.idxmax())[:7] if len(d) else "N/A"

    result = {
        "latest_growth_pct": latest_growth,
        "avg_monthly":       avg_monthly,
        "volatility_pct":    volatility,
        "peak_period":       peak_period,
        "n_periods":         len(d),
        "col_used":          amt or "row count",
    }
    log_metric("TrendAnalysis", {k:v for k,v in result.items() if isinstance(v,(int,float))})
    return result


# ── 5. Full pipeline ──────────────────────────────────────────────────────────
def run_full_pipeline():
    results = {}
    results["forecast"]     = train_forecast()
    results["segmentation"] = train_segmentation()
    results["anomaly"]      = train_anomaly_detection()
    results["trend"]        = run_trend_analysis()
    log.info("Full pipeline complete.")
    return results
