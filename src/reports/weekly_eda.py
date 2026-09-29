# -*- coding: utf-8 -*-
"""
Weekly EDA & Validation (보고서용 도표/표 자동 생성)

생성물:
- Figures (reports/figures/weekly_eda/)
  1) coverage_heatmap_top40.png         : 주간 커버리지 히트맵(품목×주)
  2) price_boxplot_top30.png            : 품목별 주간 단가 박스플롯(Top30)
  3) qty_vs_price_scatter_all.png       : 거래량-단가 산점도(전체)
  4) qty_vs_price_scatter_top{N}_*.png  : 품목별 산점도(상위 N개, 파일 다수)
  5) trends_amount_week_item_*.png      : 품목별 총액 추이
  6) trends_qty_week_item_*.png         : 품목별 반입량 추이
  7) trends_price_week_item_*.png       : 품목별 단가 추이

- Tables (reports/tables/weekly_eda/)
  a) weekly_alignment.csv               : week(월요일 0시 정렬) 정합도
  b) zero_negative_ratio.csv            : qty/amount 0·음수 비율
  c) unit_price_notna_ratio.csv         : 품목별 유효 단가 비율
  d) seasonal_gap_weeks.csv             : 품목별 최대 연속 결측 주 길이
  e) item_coverage_summary.csv          : 품목 수/주 수/기간 요약

주의: 그림이 많아질 수 있으니 기본 상위 N(거래 많은 품목)만 개별 도표 생성.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트(있으면 적용)
try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

from src.config import WEEKLY_AGG_PQ, ensure_dirs

FIG_DIR = Path("reports/figures/weekly_eda")
TAB_DIR = Path("reports/tables/weekly_eda")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# 개별 품목 그림 생성 개수(거래량 상위)
TOP_N_ITEMS_FOR_PER_ITEM_PLOTS = 12

def weekly_alignment_checks(df: pd.DataFrame) -> pd.DataFrame:
    # week가 월요일 0시(weekday==0)인지, 시간 00:00:00인지 검사
    w = pd.to_datetime(df["week"])
    align_monday = (w.dt.weekday == 0).mean()
    align_midnight = (w.dt.hour.eq(0) & w.dt.minute.eq(0) & w.dt.second.eq(0)).mean()
    out = pd.DataFrame([{
        "share_week_monday": float(align_monday),
        "share_week_midnight": float(align_midnight),
        "n_rows": int(len(w))
    }])
    out.to_csv(TAB_DIR / "weekly_alignment.csv", index=False, encoding="utf-8-sig")
    return out

def zero_negative_ratios(df: pd.DataFrame) -> pd.DataFrame:
    # qty/amount의 0 또는 음수 비율
    z_qty = (df["qty_in_week"] <= 0).mean()
    z_amt = (df["amount_week"] <= 0).mean()
    out = pd.DataFrame([{
        "ratio_qty_le_zero": float(z_qty),
        "ratio_amount_le_zero": float(z_amt),
        "n_rows": int(len(df))
    }])
    out.to_csv(TAB_DIR / "zero_negative_ratio.csv", index=False, encoding="utf-8-sig")
    return out

def unit_price_notna_by_item(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby("item")["unit_price_week"]
           .apply(lambda s: s.notna().mean())
           .reset_index(name="unit_price_notna_ratio")
           .sort_values("unit_price_notna_ratio", ascending=False))
    g.to_csv(TAB_DIR / "unit_price_notna_ratio.csv", index=False, encoding="utf-8-sig")
    return g

def max_seasonal_gap_weeks(df: pd.DataFrame) -> pd.DataFrame:
    # 품목별 unit_price_week 결측 시퀀스의 최대 연속 길이
    out_rows = []
    for it, g in df.sort_values("week").groupby("item"):
        is_na = g["unit_price_week"].isna().astype(int).values
        max_gap = 0; cur = 0
        for v in is_na:
            if v == 1:
                cur += 1
                max_gap = max(max_gap, cur)
            else:
                cur = 0
        out_rows.append({"item": it, "max_consecutive_na_weeks": int(max_gap), "n_weeks": int(len(g))})
    out = pd.DataFrame(out_rows).sort_values("max_consecutive_na_weeks", ascending=False)
    out.to_csv(TAB_DIR / "seasonal_gap_weeks.csv", index=False, encoding="utf-8-sig")
    return out

def item_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 품목 수, 주 수, 기간 범위
    weeks = pd.to_datetime(df["week"])
    out = pd.DataFrame([{
        "n_items": int(df["item"].nunique()),
        "week_min": str(weeks.min().date()) if len(weeks) else None,
        "week_max": str(weeks.max().date()) if len(weeks) else None,
        "n_weeks_total": int(len(weeks.unique())),
        "n_rows": int(len(df))
    }])
    out.to_csv(TAB_DIR / "item_coverage_summary.csv", index=False, encoding="utf-8-sig")
    return out

def plot_weekly_coverage_heatmap(df: pd.DataFrame, top_k: int = 40):
    # 품목×주 커버리지(0/1)
    d = df.dropna(subset=["item", "week"]).copy()
    d["presence"] = d["unit_price_week"].notna().astype(int)
    # 상위 품목 선택(행 수 많은 순)
    top_items = (d.groupby("item")["week"].count()
                   .sort_values(ascending=False).head(top_k).index.tolist())
    d = d[d["item"].isin(top_items)]
    mat = (d.pivot_table(index="item", columns="week", values="presence",
                         aggfunc="max", fill_value=0)
             .sort_index())
    if mat.empty:
        return
    h = max(6, 0.18 * len(mat))
    w = max(10, 0.06 * len(mat.columns))
    plt.figure(figsize=(w, h))
    plt.imshow(mat.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="coverage (0/1)")
    plt.yticks(range(len(mat.index)), mat.index)
    step = max(1, len(mat.columns)//26)
    xticks = np.arange(0, len(mat.columns), step)
    xticklabels = [str(pd.to_datetime(c).date()) for i, c in enumerate(mat.columns) if i % step == 0]
    plt.xticks(xticks, xticklabels, rotation=90)
    plt.title("주간 커버리지 히트맵 (상위 40 품목)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "coverage_heatmap_top40.png", dpi=180)
    plt.close()

def plot_price_boxplot(df: pd.DataFrame, top_k: int = 30):
    # 품목별 unit_price_week 박스플롯(Top30)
    d = df.dropna(subset=["item", "unit_price_week"])
    # 상위 품목: 관측치 많은 순
    top_items = (d.groupby("item")["unit_price_week"].size()
                   .sort_values(ascending=False).head(top_k).index.tolist())
    data = [d[d["item"] == it]["unit_price_week"].values for it in top_items]
    if not any(len(x) > 0 for x in data):
        return
    plt.figure(figsize=(max(12, 0.5 * len(top_items)), 6))
    plt.boxplot(data, showfliers=True, vert=True)
    plt.xticks(range(1, len(top_items)+1), top_items, rotation=75, ha="right")
    plt.title("품목별 주간 단가 분포(박스플롯, Top30)")
    plt.ylabel("unit_price_week")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "price_boxplot_top30.png", dpi=180)
    plt.close()

def plot_qty_price_scatter(df: pd.DataFrame):
    # 전체 산점도(알파 낮게)
    d = df.dropna(subset=["qty_in_week", "unit_price_week"])
    if d.empty: return
    plt.figure(figsize=(7.5, 6))
    plt.scatter(d["qty_in_week"], d["unit_price_week"], s=10, alpha=0.25)
    plt.xscale("symlog")  # 0 처리 안전한 로그 느낌(거래량 0~대)
    plt.yscale("symlog")
    plt.xlabel("qty_in_week")
    plt.ylabel("unit_price_week")
    plt.title("거래량-단가 산점도(전체)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "qty_vs_price_scatter_all.png", dpi=160)
    plt.close()

def plot_qty_price_scatter_by_item(df: pd.DataFrame, top_n: int = TOP_N_ITEMS_FOR_PER_ITEM_PLOTS):
    # 품목별 산점도(상위 N개)
    d = df.dropna(subset=["qty_in_week", "unit_price_week"])
    if d.empty: return
    top_items = (d.groupby("item")["qty_in_week"].sum()
                   .sort_values(ascending=False).head(top_n).index.tolist())
    for it in top_items:
        g = d[d["item"] == it]
        if g.empty: continue
        plt.figure(figsize=(6.8, 5.2))
        plt.scatter(g["qty_in_week"], g["unit_price_week"], s=16, alpha=0.6)
        plt.xscale("symlog"); plt.yscale("symlog")
        plt.xlabel("qty_in_week"); plt.ylabel("unit_price_week")
        plt.title(f"[{it}] 거래량-단가 산점도")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"qty_vs_price_scatter_{it}.png", dpi=160)
        plt.close()

def plot_trends(df: pd.DataFrame, top_n: int = TOP_N_ITEMS_FOR_PER_ITEM_PLOTS):
    # 품목별 시간 추이(총액, 반입량, 단가) — 개별 그림 3종
    # 상위 품목은 총액 합 기준
    sums = (df.groupby("item")["amount_week"].sum()
              .sort_values(ascending=False).head(top_n))
    for it in sums.index:
        g = df[df["item"] == it].sort_values("week")
        # amount_week
        plt.figure(figsize=(10, 4))
        plt.plot(g["week"], g["amount_week"])
        plt.title(f"[{it}] 주간 총액 추이")
        plt.xlabel("week"); plt.ylabel("amount_week")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"trends_amount_week_item_{it}.png", dpi=160)
        plt.close()
        # qty_in_week
        plt.figure(figsize=(10, 4))
        plt.plot(g["week"], g["qty_in_week"])
        plt.title(f"[{it}] 주간 반입량 추이")
        plt.xlabel("week"); plt.ylabel("qty_in_week")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"trends_qty_week_item_{it}.png", dpi=160)
        plt.close()
        # unit_price_week
        plt.figure(figsize=(10, 4))
        plt.plot(g["week"], g["unit_price_week"])
        plt.title(f"[{it}] 주간 단가 추이")
        plt.xlabel("week"); plt.ylabel("unit_price_week")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"trends_price_week_item_{it}.png", dpi=160)
        plt.close()

def main():
    ensure_dirs()
    df = pd.read_parquet(WEEKLY_AGG_PQ)
    df["week"] = pd.to_datetime(df["week"])

    # 체크리스트 표 생성
    weekly_alignment_checks(df)
    zero_negative_ratios(df)
    unit_price_notna_by_item(df)
    max_seasonal_gap_weeks(df)
    item_coverage_summary(df)

    # 시각화 생성
    plot_weekly_coverage_heatmap(df, top_k=40)
    plot_price_boxplot(df, top_k=30)
    plot_qty_price_scatter(df)
    plot_qty_price_scatter_by_item(df, top_n=TOP_N_ITEMS_FOR_PER_ITEM_PLOTS)
    plot_trends(df, top_n=TOP_N_ITEMS_FOR_PER_ITEM_PLOTS)

    print("[OK] Weekly EDA figures/tables generated.")
    print(" - Figures:", FIG_DIR)
    print(" - Tables :", TAB_DIR)

if __name__ == "__main__":
    main()
