# src/qa/checkpoints_simple.py
import os, pandas as pd, numpy as np
from datetime import datetime

# ============== CONFIG ==============
STAGES = [
    # (stage_name, before_path or None, after_path, index_cols)
    ("merge",   None, "data/processed/master_raw.csv", ["date","item"]),
    ("weekly",  "data/processed/master_raw.csv", "data/processed/weekly_agg.parquet", ["item","week"]),
    ("modeling",None, "data/features/weekly_features.parquet", ["item","week"]),
]
OUT_DIR = "reports/qa"
# ====================================

def _read_any(p):
    if p.endswith(".parquet"): return pd.read_parquet(p)
    if p.endswith(".csv"): return pd.read_csv(p, parse_dates=[c for c in ["date","week"] if c in open(p, "r", encoding="utf-8", errors="ignore").readline()])
    raise ValueError(f"Unsupported: {p}")

def profile(df, index_cols):
    info = {
        "rows": len(df), "cols": df.shape[1],
        "dtypes": df.dtypes.astype(str),
        "null_rate": df.isna().mean().round(4)
    }
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        info["num_summary"] = num.describe(percentiles=[.01,.25,.5,.75,.99]).T
    if index_cols:
        info["dup_rate_on_index"] = 1.0 - df[index_cols].drop_duplicates().shape[0] / max(len(df),1)
    return info

def save_html(info, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    parts = []
    parts += [f"<h2>QA Report</h2><p>generated: {datetime.now()}</p>"]
    parts += [f"<h3>Shape</h3><p>rows={info['rows']}, cols={info['cols']}</p>"]
    parts += ["<h3>Dtypes</h3>"+info["dtypes"].to_frame("dtype").to_html()]
    parts += ["<h3>Null rate</h3>"+info["null_rate"].to_frame("null_rate").to_html()]
    if "num_summary" in info:
        parts += ["<h3>Numeric summary</h3>"+info["num_summary"].to_html()]
    if "dup_rate_on_index" in info:
        parts += [f"<h3>Duplicate on index</h3><p>{info['dup_rate_on_index']:.4f}</p>"]
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(parts))

def compare(before, after):
    rows=[["rows_before",len(before)],["rows_after",len(after)],["row_change",len(after)-len(before)]]
    common = list(set(before.columns) & set(after.columns))
    for c in sorted(common):
        if pd.api.types.is_numeric_dtype(before[c]) and pd.api.types.is_numeric_dtype(after[c]):
            b, a = before[c].dropna(), after[c].dropna()
            rows += [[f"{c}.min_before", b.min() if len(b) else np.nan],
                     [f"{c}.min_after",  a.min() if len(a) else np.nan],
                     [f"{c}.max_before", b.max() if len(b) else np.nan],
                     [f"{c}.max_after",  a.max() if len(a) else np.nan],
                     [f"{c}.null_before", before[c].isna().mean()],
                     [f"{c}.null_after",  after[c].isna().mean()]]
    return pd.DataFrame(rows, columns=["metric","value"])

def run_one(stage, before_path, after_path, idx_cols):
    after = _read_any(after_path)
    info = profile(after, idx_cols)
    out_html = os.path.join(OUT_DIR, f"QA_{stage}.html")
    save_html(info, out_html)
    print("[OK] saved:", out_html)
    if before_path:
        before = _read_any(before_path)
        diff = compare(before, after)
        diff_path = os.path.join(OUT_DIR, f"QA_{stage}_diff.csv")
        diff.to_csv(diff_path, index=False)
        print("[OK] saved:", diff_path)

def main():
    for stage, bp, ap, keys in STAGES:
        if not os.path.exists(ap):
            print(f"[SKIP] {stage}: {ap} not found"); continue
        run_one(stage, bp if (bp and os.path.exists(bp)) else None, ap, keys)

if __name__ == "__main__":
    main()
