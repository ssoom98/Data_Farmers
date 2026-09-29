# -*- coding: utf-8 -*-
"""
모델 해석: LightGBM 피처 중요도 (Gain) / Permutation / (옵션) SHAP
"""
from pathlib import Path
import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

try:
    import koreanize_matplotlib  # noqa
except Exception:
    pass

from src.config import WEEKLY_FEAT_PQ, GLOBAL_LGBM_PKL, ensure_dirs

FIG_DIR = Path("reports/figures/interpret"); FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = Path("reports/tables/interpret");  TAB_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_week_log_price"
DROP_COLS = ["next_week_unit_price", TARGET, "week", "unit_price_week"]

def load_data():
    df = pd.read_parquet(WEEKLY_FEAT_PQ)
    df["week"] = pd.to_datetime(df["week"])
    feats = [c for c in df.columns if c not in DROP_COLS]
    use = df.dropna(subset=[TARGET]).copy()
    X, y = use[feats], use[TARGET]
    if "item" in X.columns:
        X["item"] = X["item"].astype("category")
    return X, y, feats

def plot_gain_importance(model):
    names = model.feature_name()
    gains = model.feature_importance(importance_type="gain")
    imp = (pd.DataFrame({"feature": names, "gain": gains})
             .sort_values("gain", ascending=False))
    top = imp.head(30)
    plt.figure(figsize=(8, max(6, 0.32*len(top))))
    plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.title("LightGBM Feature Importance (Gain) - Top 30")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_gain.png", dpi=180)
    plt.close()

def plot_permutation_importance(model, X, y):
    # 모델은 로그 스케일 타깃을 예측함
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
             .sort_values("importance", ascending=False))
    imp.to_csv(TAB_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    top = imp.head(30)
    plt.figure(figsize=(8, max(6, 0.32*len(top))))
    plt.barh(top["feature"][::-1], top["importance"][::-1], xerr=top["std"][::-1])
    plt.title("Permutation Importance (RMSE↑) - Top 30")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance_top30.png", dpi=180)
    plt.close()

def try_plot_shap(model, X):
    # SHAP 설치된 경우에만 시도
    try:
        import shap
    except Exception:
        print("[INFO] shap 미설치 → SHAP 요약은 생략")
        return
    try:
        explainer = shap.TreeExplainer(model)
        # 샘플 크기 제한(큰 데이터 대비)
        Xs = X.sample(n=min(5000, len(X)), random_state=42)
        shap_values = explainer.shap_values(Xs)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, Xs, show=False)
        plt.title("SHAP Summary (sampled)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_summary.png", dpi=180)
        plt.close()
    except Exception as e:
        print("[WARN] SHAP 실패:", e)

def main():
    ensure_dirs()
    X, y, feats = load_data()
    with open(GLOBAL_LGBM_PKL, "rb") as f:
        model = pickle.load(f)

    plot_gain_importance(model)
    # LightGBM Booster는 scikit-learn API 없이도 permutation_importance에 사용 가능(predict 호출)
    plot_permutation_importance(model, X, y)
    try_plot_shap(model, X)

    print("[OK] 모델 해석 도표 저장 →", FIG_DIR)

if __name__ == "__main__":
    main()
