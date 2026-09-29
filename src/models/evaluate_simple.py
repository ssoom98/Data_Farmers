# src/models/evaluate_simple.py
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

# ===== CONFIG =====
VAL_DIR = "predictions/val"                 # 예: snaive_val.csv, global_lgbm_val.csv ...
TRUTH_PATH = "data/processed/weekly_agg.parquet"
OUT_DIR = "reports"
# ===================

def wape(y, yhat, w=None):
    if w is None: w = np.ones_like(y, float)
    return float(np.sum(np.abs(y-yhat)*w) / np.sum(np.abs(y)*w))

def smape(y, yhat, eps=1e-9):
    return float(np.mean(2*np.abs(yhat-y)/(np.abs(y)+np.abs(yhat)+eps)))

def rmse_log(y, yhat, eps=1e-9):
    return float(np.sqrt(np.mean((np.log1p(y+eps)-np.log1p(yhat+eps))**2)))

def plot_item(df, item, out):
    d = df.sort_values("week")
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(d["week"], d["y"], label="실측"); ax.plot(d["week"], d["yhat"], label="예측")
    ax.set_title(f"[예측 vs 실측] {item}"); ax.legend()
    fig.tight_layout(); os.makedirs(out, exist_ok=True); fig.savefig(os.path.join(out, f"{item}_pred_vs_true.png")); plt.close(fig)

def main():
    truth = pd.read_parquet(TRUTH_PATH)[["item","week","unit_price_week","qty_in_week"]].rename(columns={"unit_price_week":"y","qty_in_week":"w"})
    preds = [os.path.join(VAL_DIR,f) for f in os.listdir(VAL_DIR) if f.endswith(".csv")]
    if not preds: raise SystemExit("No prediction CSVs in predictions/val")

    rows=[]; big=[]
    for p in preds:
        m = os.path.splitext(os.path.basename(p))[0]
        dfp = pd.read_csv(p, parse_dates=["week"])
        dfm = dfp.merge(truth, on=["item","week"], how="inner")
        rows.append([m, wape(dfm["y"], dfm["yhat"], dfm["w"]), smape(dfm["y"], dfm["yhat"]), rmse_log(dfm["y"], dfm["yhat"])])
        dfm["model"]=m; big.append(dfm)
    metrics = pd.DataFrame(rows, columns=["model","WAPE","sMAPE","RMSE_log"]).sort_values("WAPE")
    os.makedirs(os.path.join(OUT_DIR,"metrics"), exist_ok=True)
    metrics.to_csv(os.path.join(OUT_DIR,"metrics","baseline_val_metrics.csv"), index=False)

    big=pd.concat(big, ignore_index=True)
    by=[]; 
    for (it,m), g in big.groupby(["item","model"]):
        by.append([it,m, wape(g["y"],g["yhat"],g["w"]), smape(g["y"],g["yhat"]), rmse_log(g["y"],g["yhat"])])
    best = pd.DataFrame(by, columns=["item","model","WAPE","sMAPE","RMSE_log"]).sort_values(["item","WAPE"]).groupby("item").head(1)
    best.to_csv(os.path.join(OUT_DIR,"metrics","best_by_item.csv"), index=False)

    out_fig = os.path.join(OUT_DIR,"figures","pred_vs_true")
    for it in best["item"].head(6):
        m = best.loc[best["item"]==it, "model"].iloc[0]
        plot_item(big[(big["item"]==it)&(big["model"]==m)][["item","week","y","yhat"]], it, out_fig)

    print("[OK] saved metrics & figures to", OUT_DIR)

if __name__ == "__main__":
    main()
