# -*- coding: utf-8 -*-
"""
베이스라인(SNaive) vs GBM 검증 비교 리더보드 (견고한 컬럼 자동매핑)
입력:
 - predictions/val/snaive_val.csv    (예: item, week, y_true, y_pred)
 - predictions/val/global_lgbm_val.csv (예: item, week, next_week_unit_price, pred_unit_price)
출력:
 - reports/metrics/model_comparison_overall.csv
 - reports/metrics/model_comparison_by_item.csv
"""
import pandas as pd, numpy as np
from pathlib import Path
from src.config import PRED_VAL_DIR, REPORTS_MET_DIR, ensure_dirs

def wape(y, yhat):
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum()/denom) if denom>0 else np.nan

def smape(y, yhat):
    return float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-9)))

def _auto_map_cols(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    kind: 'snaive' or 'gbm'
    - snaive 파일에서 정답/예측 컬럼 자동감지: y_true | next_week_unit_price / y_pred | pred_unit_price
    - gbm 파일도 동일
    반환: [item, week, y_true, y_pred_*] 형태
    """
    df = df.copy()
    # 공통 기본 키
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"], errors="coerce")
    # 정답 후보
    y_true_candidates = ["y_true", "next_week_unit_price", "target", "true"]
    # 예측 후보
    if kind == "snaive":
        y_pred_candidates = ["y_pred", "pred", "pred_unit_price", "yhat"]
        pred_col_name = "y_pred_snaive"
    else:
        y_pred_candidates = ["y_pred_gbm", "pred_unit_price", "y_pred", "pred", "yhat"]
        pred_col_name = "y_pred_gbm"

    y_true_col = next((c for c in y_true_candidates if c in df.columns), None)
    y_pred_col = next((c for c in y_pred_candidates if c in df.columns), None)

    if y_true_col is None:
        raise KeyError(f"[{kind}] 정답 컬럼을 찾을 수 없습니다. 후보: {y_true_candidates}, 실제: {list(df.columns)}")
    if y_pred_col is None:
        raise KeyError(f"[{kind}] 예측 컬럼을 찾을 수 없습니다. 후보: {y_pred_candidates}, 실제: {list(df.columns)}")

    out = df.rename(columns={y_true_col: "y_true", y_pred_col: pred_col_name})
    return out[["item","week","y_true", pred_col_name]]

def main():
    ensure_dirs()
    # 파일 로드
    snaive_path = Path(PRED_VAL_DIR,"snaive_val.csv")
    gbm_path    = Path(PRED_VAL_DIR,"global_lgbm_val.csv")

    snaive = pd.read_csv(snaive_path, parse_dates=["week"], encoding="utf-8-sig")
    gbm    = pd.read_csv(gbm_path,    parse_dates=["week"], encoding="utf-8-sig")

    # 자동 매핑
    snaive_m = _auto_map_cols(snaive, "snaive")
    gbm_m    = _auto_map_cols(gbm, "gbm")

    # 병합
    df = snaive_m.merge(gbm_m, on=["item","week","y_true"], how="inner")

    # 유효행만
    df = df.dropna(subset=["y_true","y_pred_snaive","y_pred_gbm"])

    # 전체 비교
    overall = pd.DataFrame([
        {"model":"SNaive", "WAPE": wape(df["y_true"], df["y_pred_snaive"]), "sMAPE": smape(df["y_true"], df["y_pred_snaive"])},
        {"model":"GBM",    "WAPE": wape(df["y_true"], df["y_pred_gbm"]),    "sMAPE": smape(df["y_true"], df["y_pred_gbm"])}
    ])
    base_wape = overall.loc[overall["model"]=="SNaive","WAPE"].values[0]
    overall["improvement_WAPE_vs_SNaive(%)"] = np.where(
        overall["model"]=="GBM",
        (base_wape - overall["WAPE"])/base_wape * 100,
        np.nan
    )
    REPORTS_MET_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(Path(REPORTS_MET_DIR,"model_comparison_overall.csv"), index=False, encoding="utf-8-sig")

    # 품목별 비교
    def agg(g):
        return pd.Series({
            "WAPE_SNaive": wape(g["y_true"], g["y_pred_snaive"]),
            "WAPE_GBM":    wape(g["y_true"], g["y_pred_gbm"]),
            "sMAPE_SNaive": smape(g["y_true"], g["y_pred_snaive"]),
            "sMAPE_GBM":    smape(g["y_true"], g["y_pred_gbm"]),
            "n": int(len(g))
        })
    by_item = df.groupby("item").apply(agg).reset_index()
    by_item["improve_WAPE_pct"] = (by_item["WAPE_SNaive"] - by_item["WAPE_GBM"]) / by_item["WAPE_SNaive"] * 100
    by_item = by_item.sort_values("improve_WAPE_pct", ascending=False)
    by_item.to_csv(Path(REPORTS_MET_DIR,"model_comparison_by_item.csv"), index=False, encoding="utf-8-sig")

    print("[OK] 비교 저장 →",
          Path(REPORTS_MET_DIR,"model_comparison_overall.csv"),
          Path(REPORTS_MET_DIR,"model_comparison_by_item.csv"), sep="\n - ")

if __name__ == "__main__":
    main()
