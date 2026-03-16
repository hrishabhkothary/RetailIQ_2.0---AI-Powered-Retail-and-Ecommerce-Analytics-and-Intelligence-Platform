# -*- coding: utf-8 -*-
"""RetailIQ2 — Flask Backend (Schema-Independent Analytics Platform)"""
import sys, io, os, logging, threading

def _find_root():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isfile(os.path.join(cur,"requirements.txt")): return cur
        p = os.path.dirname(cur)
        if p == cur: break
        cur = p
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.chdir(ROOT)

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout,"buffer") and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr,"buffer") and not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception: pass

os.makedirs(os.path.join(ROOT,"logs"),    exist_ok=True)
os.makedirs(os.path.join(ROOT,"exports"), exist_ok=True)
os.makedirs(os.path.join(ROOT,"uploads"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(ROOT,"logs","app.log")),
              logging.StreamHandler()]
)
log = logging.getLogger(__name__)

from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime as _dt
import io as _io

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
CORS(app)

from core.data_store import store
import analytics.engine as ae
import ml.pipeline as ml

# ── Helpers ───────────────────────────────────────────────────────────────────
def ok(data=None, **kw):
    r = {"status":"ok"}
    if data is not None: r["data"] = data
    r.update(kw)
    return jsonify(r)

def err(msg, code=400):
    return jsonify({"status":"error","message":str(msg)}), code

def safe_json(obj):
    """Recursively make obj JSON-serialisable."""
    if isinstance(obj, dict):   return {k: safe_json(v) for k,v in obj.items()}
    if isinstance(obj, list):   return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    if isinstance(obj, pd.Timestamp): return str(obj)[:19]
    if isinstance(obj, float) and (obj != obj or obj == float('inf')): return None
    return obj

# ── Frontend ──────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def health(): return ok({"service":"RetailIQ2","version":"1.0","time":str(_dt.now()),"has_data":store.has_data})

# ── Dataset management ────────────────────────────────────────────────────────
@app.route("/api/datasets")
def api_datasets(): return ok(store.list_datasets())

@app.route("/api/datasets/active", methods=["POST"])
def api_set_active():
    name = (request.get_json() or {}).get("name","")
    if store.set_active(name): return ok(message=f"Active set to '{name}'")
    return err(f"Dataset '{name}' not found")

@app.route("/api/datasets/<name>", methods=["DELETE"])
def api_delete_dataset(name):
    if store.delete(name): return ok(message=f"Deleted '{name}'")
    return err(f"Dataset '{name}' not found")

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files: return err("No file sent. Use multipart key 'file'.")
    f = request.files["file"]
    fname = f.filename or "upload"
    ext = fname.rsplit(".",1)[-1].lower() if "." in fname else "csv"

    try:
        if ext == "csv":
            content = f.read()
            # try multiple encodings
            for enc in ["utf-8","latin-1","cp1252"]:
                try:
                    df = pd.read_csv(_io.BytesIO(content), encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError: continue
        elif ext in ["xlsx","xls"]:
            df = pd.read_excel(_io.BytesIO(f.read()))
        elif ext == "json":
            df = pd.read_json(_io.BytesIO(f.read()))
        elif ext == "parquet":
            df = pd.read_parquet(_io.BytesIO(f.read()))
        elif ext == "tsv":
            df = pd.read_csv(_io.BytesIO(f.read()), sep="\t", low_memory=False)
        else:
            return err(f"Unsupported format '{ext}'. Use CSV, XLSX, JSON, Parquet, or TSV.")
    except Exception as e:
        return err(f"Could not parse file: {e}")

    if df.empty or len(df.columns) == 0:
        return err("File appears empty or has no columns.")

    name = fname.rsplit(".",1)[0] if "." in fname else fname
    schema = store.ingest(df, name, fname)

    return ok({
        "name":           name,
        "rows":           schema["n_rows"],
        "cols":           schema["n_cols"],
        "primary_date":   schema.get("primary_date"),
        "primary_amount": schema.get("primary_amount"),
        "primary_qty":    schema.get("primary_qty"),
        "primary_cat":    schema.get("primary_cat"),
        "date_cols":      schema.get("date_cols",[]),
        "amount_cols":    schema.get("amount_cols",[]),
        "cat_cols":       schema.get("cat_cols",[]),
        "col_types":      schema.get("col_types",{}),
    }, message=f"Uploaded {schema['n_rows']:,} rows × {schema['n_cols']} columns")

# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/api/kpis")
def api_kpis():
    try: return jsonify({"status":"ok","data":safe_json(ae.get_kpis())})
    except Exception as e: return err(str(e),500)

@app.route("/api/kpis/filtered")
def api_kpis_filtered():
    try:
        return jsonify({"status":"ok","data":safe_json(ae.get_filtered_kpis(
            date_from=request.args.get("date_from",""),
            date_to=request.args.get("date_to",""),
            cat_col=request.args.get("cat_col",""),
            cat_val=request.args.get("cat_val",""),
        ))})
    except Exception as e: return err(str(e),500)

@app.route("/api/timeseries")
def api_timeseries():
    freq = request.args.get("freq","M")
    try: return jsonify({"status":"ok","data":safe_json(ae.get_time_series(freq))})
    except Exception as e: return err(str(e),500)

@app.route("/api/daily-trend")
def api_daily():
    days = int(request.args.get("days",90))
    try: return jsonify({"status":"ok","data":safe_json(ae.get_daily_trend(days))})
    except Exception as e: return err(str(e),500)

@app.route("/api/category")
def api_category():
    try:
        cat_col = request.args.get("cat_col","")
        val_col = request.args.get("val_col","")
        top_n   = int(request.args.get("top_n",15))
        if cat_col:
            data = ae.get_category_breakdown_by(cat_col, val_col or None, top_n)
        else:
            data = ae.get_category_breakdown(top_n=top_n)
        return jsonify({"status":"ok","data":safe_json(data)})
    except Exception as e: return err(str(e),500)

@app.route("/api/category-cols")
def api_category_cols():
    try: return jsonify({"status":"ok","data":ae.get_all_category_cols()})
    except Exception as e: return err(str(e),500)

@app.route("/api/top-n")
def api_top_n():
    try:
        return jsonify({"status":"ok","data":safe_json(ae.get_top_n(
            group_col=request.args.get("group_col",""),
            val_col=request.args.get("val_col",""),
            n=int(request.args.get("n",20)),
        ))})
    except Exception as e: return err(str(e),500)

@app.route("/api/distribution")
def api_dist():
    try:
        return jsonify({"status":"ok","data":safe_json(ae.get_numeric_distribution(
            col=request.args.get("col",""),
            bins=int(request.args.get("bins",20)),
        ))})
    except Exception as e: return err(str(e),500)

@app.route("/api/correlation")
def api_corr():
    try: return jsonify({"status":"ok","data":safe_json(ae.get_correlation_matrix())})
    except Exception as e: return err(str(e),500)

@app.route("/api/profile")
def api_profile():
    try: return jsonify({"status":"ok","data":safe_json(ae.get_data_profile())})
    except Exception as e: return err(str(e),500)

@app.route("/api/schema")
def api_schema():
    try: return jsonify({"status":"ok","data":safe_json(ae.get_schema_summary())})
    except Exception as e: return err(str(e),500)

@app.route("/api/growth")
def api_growth():
    try:
        return jsonify({"status":"ok","data":safe_json(ae.get_growth_table(
            freq=request.args.get("freq","M")
        ))})
    except Exception as e: return err(str(e),500)

@app.route("/api/outliers")
def api_outliers():
    try:
        return jsonify({"status":"ok","data":safe_json(ae.get_outliers(
            col=request.args.get("col",""),
            method=request.args.get("method","iqr"),
        ))})
    except Exception as e: return err(str(e),500)

# ── ML ────────────────────────────────────────────────────────────────────────
@app.route("/api/ml/forecast", methods=["POST"])
def api_forecast():
    try: return jsonify({"status":"ok","data":safe_json(ml.train_forecast())})
    except Exception as e: return err(str(e),500)

@app.route("/api/ml/segmentation", methods=["POST"])
def api_segmentation():
    try:
        k = int((request.get_json() or {}).get("k",5))
        return jsonify({"status":"ok","data":safe_json(ml.train_segmentation(k))})
    except Exception as e: return err(str(e),500)

@app.route("/api/ml/anomaly", methods=["POST"])
def api_anomaly():
    try:
        c = float((request.get_json() or {}).get("contamination",0.05))
        return jsonify({"status":"ok","data":safe_json(ml.train_anomaly_detection(c))})
    except Exception as e: return err(str(e),500)

@app.route("/api/ml/trend", methods=["POST"])
def api_trend():
    try: return jsonify({"status":"ok","data":safe_json(ml.run_trend_analysis())})
    except Exception as e: return err(str(e),500)

@app.route("/api/ml/pipeline", methods=["POST"])
def api_pipeline():
    try:
        def run():
            try: ml.run_full_pipeline()
            except Exception as e: log.error("Pipeline error: %s",e)
        threading.Thread(target=run, daemon=True).start()
        return jsonify({"status":"ok","message":"Full ML pipeline started in background. Check ML Center in 2-5 minutes."})
    except Exception as e: return err(str(e),500)

@app.route("/api/ml/metrics")
def api_metrics():
    try: return jsonify({"status":"ok","data":safe_json(ml.get_all_metrics())})
    except Exception as e: return err(str(e),500)

# ── Export ────────────────────────────────────────────────────────────────────
@app.route("/api/export/csv")
def api_export_csv():
    df = store.get_df()
    if df.empty: return err("No data loaded.")
    buf = _io.BytesIO(df.to_csv(index=False).encode())
    name = (store.active or "export").replace(" ","_")
    return send_file(buf, as_attachment=True, download_name=f"{name}_export.csv", mimetype="text/csv")

@app.route("/api/export/pdf", methods=["POST"])
def api_export_pdf():
    import tempfile, os as _os
    tmp = None
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER

        kpis    = ae.get_kpis()
        profile = ae.get_data_profile()
        cats    = ae.get_category_breakdown(top_n=10)
        topn    = ae.get_top_n(n=10)
        schema  = ae.get_schema_summary()

        fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="retailiq2_")
        _os.close(fd)
        doc = SimpleDocTemplate(tmp, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        sty  = getSampleStyleSheet()
        ACC  = colors.HexColor("#6366f1")
        GRAY = colors.HexColor("#6b7280")
        LTGR = colors.HexColor("#f0f0f5")

        h2  = ParagraphStyle("h2",fontName="Helvetica-Bold",fontSize=13,textColor=colors.HexColor("#0d1117"),spaceBefore=14,spaceAfter=8)
        sub = ParagraphStyle("sub",fontName="Helvetica",fontSize=9,textColor=GRAY,spaceAfter=12)
        ftr = ParagraphStyle("ftr",fontName="Helvetica",fontSize=7,textColor=GRAY,alignment=TA_CENTER,spaceBefore=8)

        def mk_tbl(data,widths,hcol=None):
            t=Table(data,colWidths=widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),hcol or ACC),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8.5),
                ("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,1),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LTGR]),
                ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#e0e0e0")),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            return t

        now   = _dt.now().strftime("%d %B %Y, %I:%M %p")
        dname = store.active or "Dataset"

        def fmtv(v):
            if v is None: return "N/A"
            if isinstance(v,(int,float)):
                if v >= 1e9: return f"{v/1e9:.2f}B"
                if v >= 1e6: return f"{v/1e6:.2f}M"
                if v >= 1e3: return f"{v/1e3:.1f}K"
                return f"{v:,.2f}"
            return str(v)

        story = [
            Spacer(1,0.5*cm),
            Paragraph("RetailIQ Analytics Report", sty["Title"]),
            Paragraph(f"Dataset: {dname}  |  Generated: {now}", sub),
            HRFlowable(width="100%",thickness=2,color=ACC,spaceAfter=14),
            Paragraph("Dataset Overview", h2),
            mk_tbl([
                ["Metric","Value"],
                ["Total Rows",           f"{kpis.get('total_rows',0):,}"],
                ["Total Columns",        str(kpis.get('total_columns',0))],
                ["Primary Amount Column",str(kpis.get('amount_label','N/A'))],
                ["Primary Date Column",  str(kpis.get('date_label','N/A'))],
                ["Primary Category Col", str(kpis.get('cat_label','N/A'))],
                ["Date Range",           f"{kpis.get('date_from','N/A')} → {kpis.get('date_to','N/A')}"],
                ["Total Amount",         fmtv(kpis.get('total_amount'))],
                ["Average Amount",       fmtv(kpis.get('avg_amount'))],
                ["MoM Growth",           f"{kpis.get('mom_growth','N/A')}%"],
            ],[8*cm,8*cm]),
            Spacer(1,0.4*cm),
        ]

        if cats:
            story += [
                Paragraph(f"Top Categories ({schema.get('primary_cat','category')})", h2),
                mk_tbl(
                    [["Category","Total Value","Count","Avg","% Share"]] +
                    [[str(c.get("label",""))[:28],fmtv(c.get("value")),
                      str(c.get("count","")),fmtv(c.get("average")),
                      f"{c.get('pct',0):.1f}%"] for c in cats[:10]],
                    [5*cm,3*cm,2.5*cm,3*cm,3*cm], colors.HexColor("#10b981")
                ),
                Spacer(1,0.4*cm),
            ]

        if topn:
            story += [
                Paragraph("Top Values", h2),
                mk_tbl(
                    [["Label","Value"]] + [[str(r.get("label",""))[:35],fmtv(r.get("value"))] for r in topn[:10]],
                    [10*cm,6*cm], colors.HexColor("#f59e0b")
                ),
                Spacer(1,0.4*cm),
            ]

        if profile.get("columns"):
            story += [
                Paragraph("Column Profile", h2),
                mk_tbl(
                    [["Column","Role","Nulls %","Unique","Min","Max"]] +
                    [[c["name"][:22],c["role"],f"{c['null_pct']}%",str(c['unique']),
                      str(c.get('min','—')),str(c.get('max','—'))] for c in profile["columns"][:20]],
                    [5*cm,2.2*cm,2*cm,2*cm,2.4*cm,3.4*cm], colors.HexColor("#8b5cf6")
                ),
            ]

        story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY,spaceAfter=4))
        story.append(Paragraph(f"RetailIQ2 v1.0  |  {now}  |  Schema-Independent Analytics Platform", ftr))
        doc.build(story)

        fname = f"RetailIQ_Report_{_dt.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response = send_file(tmp, as_attachment=True, download_name=fname, mimetype="application/pdf")
        @response.call_on_close
        def _del():
            try:
                if tmp and _os.path.isfile(tmp): _os.unlink(tmp)
            except Exception: pass
        return response
    except ImportError:
        return err("reportlab not installed. Run: pip install reportlab")
    except Exception as e:
        log.error("PDF error: %s",e,exc_info=True)
        return err(str(e),500)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("RetailIQ2 starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
