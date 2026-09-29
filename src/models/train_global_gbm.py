# -*- coding: utf-8 -*-
"""
Global LightGBM (log-price) - Holdout + Rolling CV + Reports

Outputs:
 - models/artifacts/global_lgbm.pkl
 - predictions/val/global_lgbm_val.csv
 - reports/metrics/global_lgbm_val_metrics.csv
 - reports/metrics/global_lgbm_val_by_item.csv
 - reports/metrics/global_lgbm_feature_importance.csv
 - predictions/cv/global_lgbm_cv_fold{K}.csv
 - reports/metrics/global_lgbm_cv_metrics.csv
 - reports/metrics/global_lgbm_cv_by_item.csv
"""
import pandas as pd, numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.config import (
    WEEKLY_FEAT_PQ, GLOBAL_LGBM_PKL,
    PRED_VAL_DIR, PRED_CV_DIR, REPORTS_MET_DIR,
    ensure_dirs
)
import lightgbm as lgb
import pickle

# --------------------------
# Metrics
# --------------------------
def wape(y, yhat) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum()/denom) if denom > 0 else np.nan

def smape(y, yhat) -> float:
    return float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-9))) if len(y) > 0 else np.nan

def mae(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat))) if len(y) > 0 else np.nan

def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat)**2))) if len(y) > 0 else np.nan

# --------------------------
# Splits
# --------------------------
def time_split(df: pd.DataFrame,
               train_end="2022-12-31",
               val_start="2023-01-01",
               val_end="2023-12-31") -> Tuple[pd.DataFrame, pd.DataFrame]:
    msk_tr = (df["week"] <= pd.Timestamp(train_end))
    msk_va = (df["week"] >= pd.Timestamp(val_start)) & (df["week"] <= pd.Timestamp(val_end))
    return df[msk_tr].copy(), df[msk_va].copy()

def build_rolling_year_cv(df: pd.DataFrame,
                          start_year: Optional[int] = None,
                          end_year: Optional[int] = None,
                          val_months: int = 12,
                          gap_weeks: int = 0) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    연도 단위 Rolling/Blocked CV 폴드 구성.
    각 폴드는: (train_end, val_start, val_end).
    - train: week <= train_end
    - gap: (train_end, val_start) 사이 gap_weeks 비움(정보 누설 여유)
    - val: [val_start, val_end]
    """
    w = pd.to_datetime(df["week"])
    years = sorted(w.dt.year.unique().tolist())
    if start_year is None:
        start_year = max(min(years), 2019)
    if end_year is None:
        end_year = max(years)

    folds = []
    for y in range(start_year+1, end_year+1):
        val_start = pd.Timestamp(f"{y}-01-01")
        val_end = (val_start + pd.offsets.DateOffset(months=val_months) - pd.offsets.Day(1)).normalize()
        if val_end > w.max():
            break
        train_end = val_start - pd.to_timedelta((gap_weeks+1)*7, unit="D")
        if (w <= train_end).sum() < 52:  # 최소 52주 확보
            continue
        folds.append((train_end, val_start, val_end))
    return folds

# --------------------------
# Training helpers
# --------------------------
def prepare_xy(df: pd.DataFrame, target: str):
    """
    반환: X, y, feats, use_df
    - use_df: 타깃 결측 제거 후 실제 학습/예측에 사용되는 행(저장 시 길이 매칭용)
    """
    drop_cols = ["next_week_unit_price", target, "week", "unit_price_week"]
    feats = [c for c in df.columns if c not in drop_cols]
    if "item" in df.columns:
        df["item"] = df["item"].astype("category")
    use = df.dropna(subset=[target]).copy()
    X, y = use[feats], use[target]
    return X, y, feats, use

def train_lgbm(Xtr, ytr, Xva=None, yva=None, categorical: Optional[List[str]] = None) -> lgb.Booster:
    lgb_train = lgb.Dataset(Xtr, label=ytr,
                            categorical_feature=categorical if categorical else None,
                            free_raw_data=False)
    valid_sets, valid_names = [lgb_train], ["train"]
    if Xva is not None and yva is not None:
        lgb_valid = lgb.Dataset(Xva, label=yva,
                                categorical_feature=categorical if categorical else None,
                                free_raw_data=False)
        valid_sets.append(lgb_valid)
        valid_names.append("valid")

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        feature_fraction=0.85,
        subsample=0.9,
        seed=42,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params, lgb_train,
        num_boost_round=5000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )
    return model

def eval_on_set(model: lgb.Booster, X, y_log) -> Dict[str, float]:
    """로그 스케일 타깃을 원 스케일로 복원해 평가"""
    pred_log = model.predict(X, num_iteration=model.best_iteration)
    y_true = np.expm1(y_log.values)
    y_pred = np.expm1(pred_log)
    return {
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE_log": rmse(y_log.values, pred_log)
    }

def by_item_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    # df_pred: columns = [item, week, y_true, y_pred]
    g = (df_pred.groupby("item")
         .apply(lambda g: pd.Series({
             "WAPE": wape(g["y_true"].values, g["y_pred"].values),
             "sMAPE": smape(g["y_true"].values, g["y_pred"].values),
             "MAE": mae(g["y_true"].values, g["y_pred"].values),
             "n": int(len(g))
         }))
         .reset_index()
         .sort_values("WAPE"))
    return g

# --------------------------
# Main
# --------------------------
def main():
    ensure_dirs()
    REPORTS_MET_DIR.mkdir(parents=True, exist_ok=True)
    PRED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_CV_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(WEEKLY_FEAT_PQ)
    df["week"] = pd.to_datetime(df["week"])

    target = "next_week_log_price"
    # 카테고리 유무 판단용(학습엔 각 split에서 다시 prepare_xy 호출)
    categorical = ["item"] if "item" in df.columns else None

    # ----------------------
    # Holdout (2023) 학습/평가
    # ----------------------
    tr_df, va_df = time_split(df, train_end="2022-12-31",
                              val_start="2023-01-01", val_end="2023-12-31")
    Xtr, ytr, _, tr_use = prepare_xy(tr_df, target)
    Xva, yva, _, va_use = prepare_xy(va_df, target)

    model = train_lgbm(Xtr, ytr, Xva, yva, categorical=categorical)

    # 모델 저장
    with open(GLOBAL_LGBM_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] 모델 저장 → {GLOBAL_LGBM_PKL}")

    # Holdout 예측/리포트
    va_pred_log = model.predict(Xva, num_iteration=model.best_iteration)
    va_pred = np.expm1(va_pred_log)
    y_true = np.expm1(yva.values)

    holdout_metrics = {
        "WAPE": wape(y_true, va_pred),
        "sMAPE": smape(y_true, va_pred),
        "MAE": mae(y_true, va_pred),
        "RMSE_log": rmse(yva.values, va_pred_log)
    }
    Path(REPORTS_MET_DIR, "global_lgbm_val_metrics.csv").write_text(
        "metric,value\n" + "\n".join(f"{k},{v}" for k, v in holdout_metrics.items()),
        encoding="utf-8-sig"
    )
    print("[VAL METRICS]", holdout_metrics)

    # 검증셋 예측 CSV (실측 원스케일 포함) — va_use 기준
    va_out = va_use[["item","week","next_week_unit_price"]].copy()
    va_out = va_out.rename(columns={"next_week_unit_price":"y_true"})
    va_out["y_pred"] = va_pred
    va_out.sort_values(["item","week"]).to_csv(
        Path(PRED_VAL_DIR, "global_lgbm_val.csv"),
        index=False, encoding="utf-8-sig"
    )
    # 품목별 메트릭
    by_item = by_item_metrics(va_out)
    by_item.to_csv(Path(REPORTS_MET_DIR, "global_lgbm_val_by_item.csv"),
                   index=False, encoding="utf-8-sig")

    # 피처 중요도 저장
    try:
        imp_gain = pd.DataFrame({
            "feature": model.feature_name(),
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        imp_gain.to_csv(Path(REPORTS_MET_DIR, "global_lgbm_feature_importance.csv"),
                        index=False, encoding="utf-8-sig")
    except Exception as e:
        print("[WARN] feature importance 저장 실패:", e)

    # ----------------------
    # Rolling/Blocked Yearly CV
    # ----------------------
    folds = build_rolling_year_cv(df, start_year=None, end_year=None,
                                  val_months=12, gap_weeks=0)
    print(f"[INFO] CV folds: {len(folds)}", folds)

    all_fold_preds = []
    fold_metrics = []

    for k, (train_end, val_start, val_end) in enumerate(folds, start=1):
        tr = df[df["week"] <= train_end].copy()
        va = df[(df["week"] >= val_start) & (df["week"] <= val_end)].copy()
        if tr.empty or va.empty:
            continue

        Xtr, ytr, _, tr_use_cv = prepare_xy(tr, target)
        Xva, yva, _, va_use_cv = prepare_xy(va, target)

        mdl = train_lgbm(Xtr, ytr, Xva, yva, categorical=categorical)

        pred_log = mdl.predict(Xva, num_iteration=mdl.best_iteration)
        pred = np.expm1(pred_log)
        y_t = np.expm1(yva.values)

        # 폴드 메트릭
        m = {
            "fold": k,
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "WAPE": wape(y_t, pred),
            "sMAPE": smape(y_t, pred),
            "MAE": mae(y_t, pred),
            "RMSE_log": rmse(yva.values, pred_log),
            "n": int(len(y_t))
        }
        fold_metrics.append(m)

        # 폴드 예측 저장 — va_use_cv 기준
        va_o = va_use_cv[["item","week","next_week_unit_price"]].copy()
        va_o = va_o.rename(columns={"next_week_unit_price":"y_true"})
        va_o["y_pred"] = pred
        va_o.sort_values(["item","week"]).to_csv(
            Path(PRED_CV_DIR, f"global_lgbm_cv_fold{k}.csv"),
            index=False, encoding="utf-8-sig"
        )
        va_o["fold"] = k
        all_fold_preds.append(va_o)

    # CV 전체 집계 저장
    if fold_metrics:
        cv_df = pd.DataFrame(fold_metrics)
        cv_df.to_csv(Path(REPORTS_MET_DIR, "global_lgbm_cv_metrics.csv"),
                     index=False, encoding="utf-8-sig")
        print("[CV METRICS]\n", cv_df)

        # 폴드별 예측 합쳐 품목별 메트릭 산출
        cv_preds = pd.concat(all_fold_preds, ignore_index=True)
        cv_by_item = (cv_preds.groupby("item")
                      .apply(lambda g: pd.Series({
                          "WAPE": wape(g["y_true"].values, g["y_pred"].values),
                          "sMAPE": smape(g["y_true"].values, g["y_pred"].values),
                          "MAE": mae(g["y_true"].values, g["y_pred"].values),
                          "n": int(len(g))
                      }))
                      .reset_index()
                      .sort_values("WAPE"))
        cv_by_item.to_csv(Path(REPORTS_MET_DIR, "global_lgbm_cv_by_item.csv"),
                          index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
