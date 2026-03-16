# RetailIQ 2.0 — Universal Schema-Independent Analytics Platform

> Upload **any** CSV, Excel, JSON, or Parquet file. The system auto-detects column types and instantly provides full analytics, ML forecasting, segmentation, and anomaly detection — no configuration required.

---

## Key Principle: Schema Independence

RetailIQ 2.0 works on **any structured dataset**:

| Upload This | Get This |
|-------------|----------|
| Superstore Sales CSV (Kaggle) | Revenue by category, demand forecast, customer segments |
| Bank transactions Excel | Anomaly detection on amounts, trend by month |
| E-commerce orders JSON | MoM growth, top products, inventory insight |
| Hospital billing Parquet | Spend analysis, outlier detection, time series |
| Any CSV with numbers + dates | All analytics auto-configured |

No `column_name = "Sales"` anywhere in the code. The schema detector reads your file, assigns roles (date / amount / category / quantity / profit / id), and every chart, KPI, and ML model uses those discovered roles automatically.

---

## Quick Start

```bash
# 1. Install Python 3.10+ from python.org
# 2. Extract the ZIP to a folder e.g. C:\RetailIQ2\

cd C:\RetailIQ2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py

# 5. Open browser
# http://localhost:5000

# 6. Click "Upload Data" → drag any CSV/Excel/JSON file → done
```

**No MySQL. No database setup. No config files to edit.**

---

## Architecture

```
RetailIQ2/
├── app.py                     # Flask server — 25+ API routes
├── requirements.txt
├── core/
│   ├── schema_detector.py     # Auto-detects column roles from ANY dataset
│   └── data_store.py          # In-memory store + Parquet persistence
├── analytics/
│   └── engine.py              # All BI queries — schema-independent
├── ml/
│   └── pipeline.py            # RF forecast + KMeans + IsoForest + Trend
└── ui/
    └── templates/index.html   # Full SPA dashboard (single file)
```

---

## Features

| Module | What It Does |
|--------|-------------|
| Overview Dashboard | Auto-built KPIs, monthly trend, category donut, growth chart |
| Trend Analysis | Daily chart (30/60/90/180d), top-N values, growth table |
| Category Analysis | Breakdown by any detected categorical column × any value column |
| Distribution | Histogram for any numeric column + correlation matrix |
| Demand Forecast | Random Forest — auto-discovers date + numeric, 30-period ahead |
| Segmentation | K-Means on all numeric columns, schema-independent clusters |
| Anomaly Detection | Isolation Forest on all numeric features |
| Data Profile | Column roles, null %, unique counts, min/max per column |
| PDF Export | Full styled report from live data |
| Multi-Dataset | Upload multiple files, switch active dataset |

---

## Supported Kaggle Datasets (works out-of-the-box)

- **Sample Superstore** — kaggle.com/datasets/vivek468/superstore-dataset-final
- **Brazilian E-Commerce (Olist)** — kaggle.com/datasets/olistbr/brazilian-ecommerce
- **UK E-Commerce** — kaggle.com/datasets/carrie1/ecommerce-data
- **Walmart Sales** — kaggle.com/datasets/mikhail1681/walmart-sales
- **Amazon Sales** — kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- **Global Superstore** — kaggle.com/datasets/apoorvaappz/global-super-store-dataset
- Any other sales, finance, or operations CSV

---

## Tech Stack

| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.10–3.12 | Runtime |
| Flask | 3.0.3 | Web server + REST API |
| Flask-CORS | 4.0.1 | Cross-origin headers |
| Pandas | 2.2.2 | Data loading, feature engineering, BI queries |
| NumPy | 1.26.4 | Numeric operations, type safety |
| scikit-learn | 1.4.2 | RandomForest, KMeans, IsolationForest, StandardScaler |
| statsmodels | 0.14.2 | Rolling trend analysis |
| joblib | 1.4.2 | Model serialisation (.pkl files) |
| ReportLab | 4.1.0 | PDF report generation |
| openpyxl | 3.1.2 | Excel (.xlsx) file reading |
| pyarrow | 16.1.0 | Parquet read/write for data persistence |
| Chart.js | 4.4.1 (CDN) | All dashboard charts |
| Gunicorn | 22.0.0 | Production WSGI server (Linux) |

---

## GitHub

```bash
git init && git add . && git commit -m "RetailIQ2 v1.0"
git remote add origin https://github.com/YOUR/retailiq2.git
git push -u origin main
```

`.gitignore` excludes: `uploads/store/` (data), `models/` (trained pkl), `logs/`, `exports/`

---

*Built with Python · Flask · scikit-learn · Pandas · Chart.js*
