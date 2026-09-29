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

# --- scikit-learn 호환 래퍼 ---
class BoosterWrapper:
    """lightgbm.Booster 를 sklearn estimator처럼 감싸는 래퍼"""
    def __init__(self, booster):
        self.booster = booster
        self.best_iteration_ = getattr(booster, "best_iteration", None)

    def fit(self, X, y):
        return self  # 이미 학습됨

    def predict(self, X):
        return self.booster.predict(X, num_iteration=self.best_iteration_)

    def score(self, X, y):
        yhat = self.predict(X)
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return -float(rmse)  # 클수록 좋게

def load_data():
    df = pd.read_parquet(WEEKLY_FEAT_PQ).copy()
    df["week"] = pd.to_datetime(df["week"])
    if "item" in df.columns:
        df["item"] = df["item"].astype("category")  # <- 원본에서 캐스팅

    feats = [c for c in df.columns if c not in DROP_COLS]
    use = df.dropna(subset=[TARGET]).copy()
    X = use[feats].copy()
    y = use[TARGET].copy()
    return X, y, feats

def plot_gain_importance(booster):
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    imp = (pd.DataFrame({"feature": names, "gain": gains})
             .sort_values("gain", ascending=False))
    top = imp.head(30)
    plt.figure(figsize=(8, max(6, 0.32*len(top))))
    plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.title("LightGBM Feature Importance (Gain) - Top 30")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_gain.png", dpi=180)
    plt.close()

def plot_permutation_importance(booster, X, y, sample_n=8000):
    # 큰 데이터 대비 샘플링
    if len(X) > sample_n:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X), size=sample_n, replace=False)
        Xs, ys = X.iloc[idx].copy(), y.iloc[idx].copy()
    else:
        Xs, ys = X, y

    est = BoosterWrapper(booster)
    r = permutation_importance(est, Xs, ys, n_repeats=10, random_state=42, n_jobs=-1)
    imp = (pd.DataFrame({"feature": Xs.columns, "importance": r.importances_mean, "std": r.importances_std})
             .sort_values("importance", ascending=False))
    imp.to_csv(TAB_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    top = imp.head(30)
    plt.figure(figsize=(8, max(6, 0.32*len(top))))
    plt.barh(top["feature"][::-1], top["importance"][::-1], xerr=top["std"][::-1])
    plt.title("Permutation Importance (−RMSE 기준, Top 30)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance_top30.png", dpi=180)
    plt.close()

def try_plot_shap(booster, X):
    try:
        import shap
    except Exception:
        print("[INFO] shap 미설치 → SHAP 요약은 생략")
        return
    try:
        explainer = shap.TreeExplainer(booster)
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
        booster = pickle.load(f)

    # 1) Gain importance
    plot_gain_importance(booster)

    # 2) Permutation importance (BoosterWrapper 사용)
    plot_permutation_importance(booster, X, y)

    # 3) (옵션) SHAP
    try_plot_shap(booster, X)

    print("[OK] 모델 해석 도표 저장 →", FIG_DIR)

if __name__ == "__main__":
    main()
