# -*- coding: utf-8 -*-
"""
SNaive(52주 계절) 베이스라인 생성 + 검증 성능 산출
입력:
 - data/processed/weekly_agg.parquet  (item, week, unit_price_week)
출력:
 - predictions/val/snaive_val.csv
 - reports/metrics/baseline_val_metrics.csv
 - reports/metrics/baseline_val_by_item.csv
"""
import pandas as pd, numpy as np
from pathlib import Path
from src.config import WEEKLY_AGG_PQ, PRED_VAL_DIR, REPORTS_MET_DIR, ensure_dirs

def wape(y, yhat):
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum()/denom) if denom>0 else np.nan

def smape(y, yhat):
    return float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-9)))

def main():
    ensure_dirs()
    wk = pd.read_parquet(WEEKLY_AGG_PQ)
    wk["week"] = pd.to_datetime(wk["week"])
    wk = wk.sort_values(["item","week"]).copy()

    # 타깃: 다음 주 실측
    wk["y_true_next"] = wk.groupby("item")["unit_price_week"].shift(-1)

    # (t+1) 예측 = (t+1-52) 실측을 사용 → 안전하게 조인으로 계산
    df = wk[["item","week","unit_price_week","y_true_next"]].copy()
    df["week_minus_51"] = df["week"] - pd.to_timedelta(51*7, unit="D")  # (t+1)-52 == t-51
    past = wk.rename(columns={"week":"week_minus_51", "unit_price_week":"yhat_snaive_src"})[["item","week_minus_51","yhat_snaive_src"]]
    df = df.merge(past, on=["item","week_minus_51"], how="left").drop(columns=["week_minus_51"])

    # 검증 기간(예: 2023년)
    val = df[(df["week"]>=pd.Timestamp("2023-01-01")) & (df["week"]<=pd.Timestamp("2023-12-31"))].copy()
    val = val.rename(columns={"y_true_next":"y_true", "yhat_snaive_src":"y_pred"})
    val = val.dropna(subset=["y_true","y_pred"])

    # 지표
    overall = {
        "WAPE": wape(val["y_true"], val["y_pred"]),
        "sMAPE": smape(val["y_true"], val["y_pred"]),
        "MAE": float(np.mean(np.abs(val["y_true"]-val["y_pred"]))),
        "n": int(len(val))
    }
    by_item = (val.groupby("item")
                 .apply(lambda g: pd.Series({
                     "WAPE": wape(g["y_true"], g["y_pred"]),
                     "sMAPE": smape(g["y_true"], g["y_pred"]),
                     "MAE": float(np.mean(np.abs(g["y_true"]-g["y_pred"]))),
                     "n": int(len(g))
                 }))
                 .reset_index())

    # 저장
    PRED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_MET_DIR.mkdir(parents=True, exist_ok=True)
    val[["item","week","y_true","y_pred"]].sort_values(["item","week"]).to_csv(
        Path(PRED_VAL_DIR, "snaive_val.csv"), index=False, encoding="utf-8-sig"
    )
    Path(REPORTS_MET_DIR, "baseline_val_metrics.csv").write_text(
        "metric,value\n" + "\n".join([f"{k},{v}" for k,v in overall.items()]), encoding="utf-8-sig"
    )
    by_item.to_csv(Path(REPORTS_MET_DIR, "baseline_val_by_item.csv"), index=False, encoding="utf-8-sig")
    print("[OK] SNaive 완료 →",
          Path(PRED_VAL_DIR, "snaive_val.csv"),
          Path(REPORTS_MET_DIR, "baseline_val_metrics.csv"),
          Path(REPORTS_MET_DIR, "baseline_val_by_item.csv"), sep="\n - ")

if __name__ == "__main__":
    main()
