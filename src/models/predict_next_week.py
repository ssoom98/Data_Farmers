# -*- coding: utf-8 -*-
"""
운영용: 최신 주 기준 '다음 주' 예측 산출
입력:
 - data/features/weekly_features.parquet
 - models/artifacts/global_lgbm.pkl
출력:
 - predictions/next_week/global_lgbm_next_week.csv
"""
import pandas as pd, numpy as np, pickle
from pathlib import Path
from src.config import WEEKLY_FEAT_PQ, GLOBAL_LGBM_PKL, PRED_NEXT_DIR, ensure_dirs

def main():
    ensure_dirs()
    df = pd.read_parquet(WEEKLY_FEAT_PQ)
    df["week"] = pd.to_datetime(df["week"])

    target = "next_week_log_price"
    drop_cols = ["next_week_unit_price", target, "week", "unit_price_week"]
    feats = [c for c in df.columns if c not in drop_cols]

    # 예측 대상: 타깃 결측 (마지막 주)
    pred_cand = df[df["next_week_unit_price"].isna()].copy()
    if pred_cand.empty:
        last_week = df["week"].max()
        pred_cand = df[df["week"]==last_week].copy()

    if "item" in feats:
        pred_cand["item"] = pred_cand["item"].astype("category")

    with open(GLOBAL_LGBM_PKL, "rb") as f:
        model = pickle.load(f)

    pred_log = model.predict(pred_cand[feats], num_iteration=getattr(model,"best_iteration", None))
    pred = np.expm1(pred_log)

    out = pred_cand[["item","week"]].copy()
    out["pred_next_week_unit_price"] = pred
    PRED_NEXT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(PRED_NEXT_DIR, "global_lgbm_next_week.csv"), index=False, encoding="utf-8-sig")
    print("[OK] 다음 주 예측 저장 →", Path(PRED_NEXT_DIR, "global_lgbm_next_week.csv"))

if __name__ == "__main__":
    main()
