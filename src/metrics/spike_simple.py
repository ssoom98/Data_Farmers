# src/metrics/spike_simple.py
import os, pandas as pd, numpy as np

# ===== CONFIG =====
PRED_PATH = "predictions/val/snaive_val.csv"   # 또는 global_lgbm_val.csv 등
TRUTH_PATH = "data/processed/weekly_agg.parquet"
THETA = 0.30        # +30% 급등을 스파이크로 간주
OUT_CSV = "reports/metrics/spike_metrics.csv"
# ===================

def label_spike(df, col):
    d=df.sort_values("week").copy()
    d["ret"]=d[col].pct_change()
    d["spike"]=(d["ret"]>THETA).astype(int)
    return d

def prf1(y_true, y_pred):
    tp=((y_true==1)&(y_pred==1)).sum(); fp=((y_true==0)&(y_pred==1)).sum(); fn=((y_true==1)&(y_pred==0)).sum()
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0; rec=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def main():
    p = pd.read_csv(PRED_PATH, parse_dates=["week"])
    t = pd.read_parquet(TRUTH_PATH)[["item","week","unit_price_week"]].rename(columns={"unit_price_week":"y"})
    df = p.merge(t, on=["item","week"], how="inner")
    rows=[]
    for it,g in df.groupby("item"):
        gt = label_spike(g, "y"); pr = label_spike(g.assign(y=g["yhat"]), "y")
        prc, rec, f1 = prf1(gt["spike"].values, pr["spike"].values)
        rows.append([it, prc, rec, f1])
    out = pd.DataFrame(rows, columns=["item","precision","recall","f1"])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("[OK] saved:", OUT_CSV)

if __name__ == "__main__":
    main()
