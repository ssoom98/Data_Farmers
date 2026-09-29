# -*- coding: utf-8 -*-
"""
변동성 타임라인: price_vol_12 상위 구간(예: 상위 20%) 알람 표시
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

from src.config import WEEKLY_FEAT_PQ, ensure_dirs

FIG_DIR = Path("reports/figures/volatility"); FIG_DIR.mkdir(parents=True, exist_ok=True)
TOP_N = 12  # 그림 생성 품목 수 (amount 합 상위)

def plot_one(g: pd.DataFrame, thr: float):
    it = g["item"].iloc[0]
    plt.figure(figsize=(10, 4.2))
    plt.plot(g["week"], g["unit_price_week"], label="unit_price_week")
    # 알람 구간: price_vol_12 >= 분위수 임계치
    alarm = g["price_vol_12"] >= thr
    plt.scatter(g.loc[alarm, "week"], g.loc[alarm, "unit_price_week"], s=32, marker="^", label="high volatility")
    plt.title(f"[{it}] 단가 & 변동성 알람(≥Q80 of price_vol_12)")
    plt.xlabel("week"); plt.ylabel("unit_price_week"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / f"vol_timeline_{it}.png", dpi=170)
    plt.close()

def main():
    ensure_dirs()
    df = pd.read_parquet(WEEKLY_FEAT_PQ)
    df["week"] = pd.to_datetime(df["week"])

    # 임계치: 전체 price_vol_12의 80% 분위
    if "price_vol_12" not in df.columns:
        # make_weekly_features.py에서 생성하도록 안내
        print("[WARN] price_vol_12 없음 → 변동성 타임라인 건너뜀")
        return
    thr = float(np.nanpercentile(df["price_vol_12"].dropna(), 80)) if df["price_vol_12"].notna().any() else np.inf

    # 품목 선택: amount_week 합 상위
    top_items = (df.groupby("item")["amount_week"].sum()
                 .sort_values(ascending=False).head(TOP_N).index.tolist())
    for it in top_items:
        g = df[df["item"] == it].sort_values("week")
        plot_one(g, thr)

    print("[OK] 변동성 타임라인 저장 →", FIG_DIR)

if __name__ == "__main__":
    main()
