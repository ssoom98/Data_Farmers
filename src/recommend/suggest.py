# -*- coding: utf-8 -*-
"""
대체재 추천 v1: 동일 군 내에서 '가격 절감 + 가용성 양호 + 변동성 낮음' 기준 Top-K 추천
입력:
 - predictions/next_week/global_lgbm_next_week.csv
 - data/processed/weekly_agg.parquet
 - suggestions/substitution_rules.yaml
출력:
 - suggestions/suggestions_next_week.csv
"""
import pandas as pd, numpy as np, yaml
from pathlib import Path
from src.config import WEEKLY_AGG_PQ, PRED_NEXT_DIR, SUGG_RULES_YAML, SUGG_NEXT_CSV, ensure_dirs

def load_rules(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    groups = cfg.get("groups", {})
    rules  = cfg.get("rules", {})
    item2grp = {str(it): g for g, items in groups.items() for it in items}
    return groups, item2grp, rules

def main():
    ensure_dirs()
    # 입력 로드
    pred = pd.read_csv(Path(PRED_NEXT_DIR, "global_lgbm_next_week.csv"), parse_dates=["week"])
    pred = pred.rename(columns={"pred_next_week_unit_price":"pred_price"})
    wk = pd.read_parquet(WEEKLY_AGG_PQ)
    wk["week"] = pd.to_datetime(wk["week"])

    last_week = pred["week"].max()
    cur = wk[wk["week"]==last_week][["item","unit_price_week","qty_in_week"]].rename(
        columns={"unit_price_week":"this_week_price","qty_in_week":"this_week_qty"}
    )
    df = pred.merge(cur, on="item", how="left")

    # 최근 4주 가용성, 8주 변동성
    w = wk.sort_values(["item","week"]).copy()
    w["qty_rollmean4"] = w.groupby("item")["qty_in_week"].shift(1).rolling(4, min_periods=2).mean()
    w["ret"] = w.groupby("item")["unit_price_week"].pct_change()
    w["price_vol_8"] = w["ret"].shift(1).rolling(8, min_periods=2).std()
    df = df.merge(w[["item","week","qty_rollmean4","price_vol_8"]], on=["item","week"], how="left")

    # 룰/군
    groups, item2grp, rules = load_rules(SUGG_RULES_YAML)
    tau = float(rules.get("price_discount_threshold", 0.15))
    top_k = int(rules.get("top_k", 3))

    df["group"] = df["item"].map(item2grp).fillna("기타")
    # 군별 기준치 (같은 '현재 주'의 군 분포)
    grp_stats = (df.groupby("group")
                   .agg(supply_med=("qty_rollmean4","median"),
                        vol_q75=("price_vol_8", lambda s: np.nanpercentile(s.dropna(), 75) if s.notna().any() else np.nan))
                   .reset_index())
    df = df.merge(grp_stats, on="group", how="left")

    # 추천 생성
    out_rows = []
    for g, gdf in df.groupby("group"):
        items = gdf["item"].unique().tolist()
        supply_med = gdf["supply_med"].iloc[0]
        vol_q75 = gdf["vol_q75"].iloc[0]
        for a in items:
            pA = gdf.loc[gdf["item"]==a, "pred_price"].values[0]
            cand = gdf[gdf["item"]!=a].copy()
            cand["saving_rate"] = (pA - cand["pred_price"])/(pA+1e-9)
            cond_price = cand["saving_rate"] >= tau
            cond_supply = True if pd.isna(supply_med) else (cand["qty_rollmean4"] >= supply_med)
            cond_vol = True if pd.isna(vol_q75) else (cand["price_vol_8"] <= vol_q75)
            c2 = cand[cond_price & cond_supply & cond_vol].sort_values(
                ["saving_rate","price_vol_8"], ascending=[False, True]
            ).drop_duplicates(subset=["item"]).head(top_k)
            for _, r in c2.iterrows():
                out_rows.append({
                    "base_item": a,
                    "base_group": g,
                    "base_pred_price": pA,
                    "alt_item": r["item"],
                    "alt_pred_price": r["pred_price"],
                    "saving_rate": r["saving_rate"],
                    "alt_supply_recent4w": r["qty_rollmean4"],
                    "alt_vol_8w": r["price_vol_8"],
                    "reason": f"동일군 저가(≥{int(tau*100)}%), 공급≥군 중앙, 변동성≤군 75%"
                })

    out = pd.DataFrame(out_rows).sort_values(["base_item","saving_rate"], ascending=[True,False])
    SUGG_NEXT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUGG_NEXT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 대체재 추천 저장 → {SUGG_NEXT_CSV} (rows={len(out)})")

if __name__ == "__main__":
    main()
