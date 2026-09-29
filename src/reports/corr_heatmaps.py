# -*- coding: utf-8 -*-
"""
품목별 상관 히트맵: log_price vs 랙/롤링/기상
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

from src.config import WEEKLY_FEAT_PQ, ensure_dirs

FIG_DIR = Path("reports/figures/corr"); FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = Path("reports/tables/corr");  TAB_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "log_price"
CAND_PREFIX = ["log_price_lag", "log_price_rollmean", "log_price_rollstd",
               "log_qty_lag", "log_qty_rollmean", "log_qty_rollstd",
               "price_vol_", "tavg", "tmin", "tmax", "precip", "humidity",
               "sunshine", "radiation", "wind", "weekofyear", "month", "year"]

def select_columns(df):
    cols = []
    for c in df.columns:
        if c == TARGET: continue
        if any(c.startswith(p) for p in CAND_PREFIX) or c in ["tavg","tmin","tmax","precip","humidity","sunshine","radiation","wind"]:
            cols.append(c)
    # 숫자형만
    cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return list(dict.fromkeys(cols))  # 고유 순서 보존

def corr_by_item(df, items_top=20):
    # 관측수 많은 품목 상위 N개 선정
    order = (df.groupby("item")[TARGET].count()
               .sort_values(ascending=False).head(items_top)).index.tolist()
    feats = select_columns(df)
    rows = []
    for it in order:
        g = df[df["item"] == it]
        sub = g[[TARGET] + feats].dropna()
        if len(sub) < 10:  # 표본 너무 적으면 스킵
            continue
        corr = sub.corr(numeric_only=True)[TARGET].drop(TARGET, errors="ignore")
        for f, v in corr.items():
            rows.append({"item": it, "feature": f, "corr": float(v)})
    long = pd.DataFrame(rows)
    long.to_csv(TAB_DIR / "corr_matrix_long.csv", index=False, encoding="utf-8-sig")

    # 히트맵(특징 수가 많으니 상위 상관 절대값 큰 피처만 추려도 됨)
    # 여기서는 상위 25개 피처 기준으로 컬럼 선택
    top_feats = (long.groupby("feature")["corr"]
                     .apply(lambda s: s.abs().mean())
                     .sort_values(ascending=False).head(25).index.tolist())
    mat = (long[long["feature"].isin(top_feats)]
           .pivot_table(index="item", columns="feature", values="corr"))
    if mat.empty: return
    plt.figure(figsize=(max(10, 0.6*len(top_feats)), max(6, 0.4*len(mat))))
    plt.imshow(mat.values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=75, ha="right")
    plt.title("품목별 상관 히트맵: log_price vs 후보 피처")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corr_heatmap_item_top20.png", dpi=180)
    plt.close()

def main():
    ensure_dirs()
    df = pd.read_parquet(WEEKLY_FEAT_PQ)
    df["week"] = pd.to_datetime(df["week"])
    corr_by_item(df, items_top=20)
    print("[OK] 상관 히트맵 저장 →", FIG_DIR)

if __name__ == "__main__":
    main()
