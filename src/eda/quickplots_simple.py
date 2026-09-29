# src/eda/quickplots_simple.py
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

# ===== CONFIG =====
WEEKLY_PATH = "data/processed/weekly_agg.parquet"
OUT_DIR = "reports/figures"
MAX_ITEMS = 12           # 너무 많으면 상위 일부만
ITEM_FILTER = []         # ["시금치","감자"] 식으로 지정 가능
# ===================

def ensure(p): os.makedirs(p, exist_ok=True)

def plot_trend(df, item, out):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(df["week"], df["unit_price_week"])
    ax.set_title(f"[추세] {item} 주간 단가"); ax.set_xlabel("week"); ax.set_ylabel("unit_price_week")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{item}_trend.png")); plt.close(fig)

def plot_box_month(df, item, out):
    g = df.copy(); g["month"] = g["week"].dt.month
    fig, ax = plt.subplots(figsize=(9,4))
    g.boxplot(column="unit_price_week", by="month", ax=ax)
    ax.set_title(f"[월별 분포] {item}"); ax.set_xlabel("month"); ax.set_ylabel("unit_price_week")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(os.path.join(out, f"{item}_box_month.png")); plt.close(fig)

def plot_heat(df, item, out):
    g = df.copy()
    g["y"] = g["week"].dt.year; g["woy"] = g["week"].dt.isocalendar().week.astype(int)
    pt = g.pivot_table(index="y", columns="woy", values="unit_price_week", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11,4))
    im = ax.imshow(pt, aspect="auto", origin="lower"); ax.set_title(f"[계절성] {item} (year × weekofyear)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(os.path.join(out, f"{item}_heatmap.png")); plt.close(fig)

def main():
    df = pd.read_parquet(WEEKLY_PATH).sort_values(["item","week"])
    itms = df["item"].dropna().unique().tolist()
    if ITEM_FILTER: itms = [x for x in itms if x in ITEM_FILTER]
    itms = itms[:MAX_ITEMS]

    out_tr, out_bx, out_ht = [os.path.join(OUT_DIR, s) for s in ["eda_trend","eda_box","eda_heat"]]
    for d in [out_tr, out_bx, out_ht]: ensure(d)

    for it in itms:
        d = df[df["item"]==it]
        plot_trend(d, it, out_tr); plot_box_month(d, it, out_bx); plot_heat(d, it, out_ht)

    print("[OK] saved EDA images in:", OUT_DIR)

if __name__ == "__main__":
    main()
