# -*- coding: utf-8 -*-
"""
03_quick_review.py
- 프로젝트 산출물(원시/주간/피처/검증예측/운영예측) 자동 점검 & 리포트 저장
- CLI 없이 실행 가능. 아래 PROJECT_ROOT만 네 경로로 맞추면 됨.
"""

from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
# --- 한글 폰트 설정: Windows / macOS / Linux 자동 대응 ---
import os, sys
from matplotlib import font_manager, rcParams

def _set_korean_font():
    cand_paths = []
    if os.name == "nt":  # Windows
        # 기본 윈도우 한글 폰트
        cand_paths += [
            r"C:\Windows\Fonts\malgun.ttf",          # Malgun Gothic Regular
            r"C:\Windows\Fonts\malgunbd.ttf",        # Malgun Gothic Bold
            r"C:\Windows\Fonts\gulim.ttc",
        ]
        cand_families = ["Malgun Gothic", "Gulim"]
    elif sys.platform == "darwin":  # macOS
        cand_paths += ["/System/Library/Fonts/AppleGothic.ttf"]
        cand_families = ["AppleGothic"]
    else:  # Linux
        cand_paths += [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        cand_families = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]

    loaded_family = None
    for p in cand_paths:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                # 폰트 파일의 family명을 가져와 설정
                prop = font_manager.FontProperties(fname=p)
                loaded_family = prop.get_name()
                break
            except Exception:
                pass

    # family 우선순위: 찾은 family → 후보 리스트
    families = [loaded_family] if loaded_family else []
    families += cand_families

    # matplotlib 기본 폰트 교체
    rcParams["font.family"] = families
    rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

    # 디버그용: 실제 적용된 폰트 확인
    # print("Korean font set to:", rcParams["font.family"])

_set_korean_font()


# 0) 프로젝트 루트 (네 경로로 수정)
PROJECT_ROOT = Path(r"D:\TMD_final_agri1")

# 경로들
MASTER_RAW_CSV   = PROJECT_ROOT / "data" / "processed" / "master_raw.csv"
WEEKLY_AGG_PQ    = PROJECT_ROOT / "data" / "processed" / "weekly_agg.parquet"
WEEKLY_FEAT_PQ   = PROJECT_ROOT / "data" / "features" / "weekly_features.parquet"
VAL_PRED_CSV     = PROJECT_ROOT / "predictions" / "val" / "global_lgbm_val.csv"
NEXT_PRED_CSV    = PROJECT_ROOT / "predictions" / "next_week" / "global_lgbm_next_week.csv"

FIG_DIR          = PROJECT_ROOT / "reports" / "figures"
MET_DIR          = PROJECT_ROOT / "reports" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 유틸 ----------
def wape(y, yhat, w=None):
    if w is None:
        denom = np.abs(y).sum()
        return float(np.abs(y - yhat).sum()/denom) if denom>0 else np.nan
    denom = np.abs(y*w).sum()
    return float(np.abs(y - yhat).mul(w).abs().sum()/denom) if denom>0 else np.nan

def smape(y, yhat):
    return float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-9)))

def rmse_log(y_log, yhat_log):
    return float(np.sqrt(np.mean((y_log - yhat_log)**2)))

# 1) master_raw 검수
raw = pd.read_csv(MASTER_RAW_CSV, parse_dates=["date"])
raw["unit_price"] = pd.to_numeric(raw["unit_price"], errors="coerce")
raw["qty_in"]     = pd.to_numeric(raw["qty_in"], errors="coerce")
raw["amount"]     = pd.to_numeric(raw["amount"], errors="coerce")

summary_item = (
    raw.groupby("item")
       .agg(first_date=("date","min"),
            last_date=("date","max"),
            n_rows=("date","size"),
            zero_qty=("qty_in", lambda s: int((s<=0).sum())),
            zero_amt=("amount", lambda s: int((s<=0).sum())),
            nan_price=("unit_price", lambda s: int(s.isna().sum())),
            price_median=("unit_price","median"),
            price_iqr=("unit_price", lambda s: s.quantile(0.75)-s.quantile(0.25)))
       .reset_index()
)
summary_item.to_csv(MET_DIR/"master_raw_item_summary.csv", index=False, encoding="utf-8-sig")

# 분포 그림(상자그림) 일부 품목만(상위 12개)
top_items = summary_item.sort_values("n_rows", ascending=False)["item"].head(12).tolist()
plt.figure()
raw[raw["item"].isin(top_items)].boxplot(column="unit_price", by="item", rot=60)
plt.title("품목별 단가 분포(샘플 12)")
plt.suptitle("")
plt.ylabel("단가")
plt.tight_layout()
plt.savefig(FIG_DIR/"box_unit_price_sample12.png", dpi=150)
plt.close()

# 2) weekly_agg 검수
wk = pd.read_parquet(WEEKLY_AGG_PQ)
wk["week"] = pd.to_datetime(wk["week"])
wk_summary = (
    wk.assign(valid_price=~wk["unit_price_week"].isna())
      .groupby("item")
      .agg(weeks=("week","nunique"),
           valid_weeks=("valid_price","sum"),
           valid_ratio=("valid_price","mean"),
           start=("week","min"),
           end=("week","max"),
           qty_median=("qty_in_week","median"))
      .reset_index()
)
wk_summary.to_csv(MET_DIR/"weekly_agg_summary.csv", index=False, encoding="utf-8-sig")

# 3) weekly_features 검수
feat = pd.read_parquet(WEEKLY_FEAT_PQ)
feat["week"] = pd.to_datetime(feat["week"])
# 결측률 상위 30개 피처 보고
null_rate = feat.isna().mean().sort_values(ascending=False).head(30).rename("null_rate").to_frame()
null_rate.to_csv(MET_DIR/"feature_null_rate_top30.csv", encoding="utf-8-sig")

# 4) 검증 예측 성능
val = pd.read_csv(VAL_PRED_CSV, parse_dates=["week"])
# y_true가 next_week_unit_price 컬럼
val = val.rename(columns={"next_week_unit_price":"y_true", "pred_unit_price":"y_pred"})
val = val.dropna(subset=["y_true","y_pred"])

# 전체 지표
overall = pd.DataFrame([{
    "metric":"WAPE", "value": wape(val["y_true"], val["y_pred"])
},{
    "metric":"sMAPE", "value": smape(val["y_true"], val["y_pred"])
}])
overall.to_csv(MET_DIR/"val_overall_metrics.csv", index=False, encoding="utf-8-sig")

# 품목별 지표
by_item = (val.groupby("item")
             .apply(lambda g: pd.Series({
                 "WAPE": wape(g["y_true"], g["y_pred"]),
                 "sMAPE": smape(g["y_true"], g["y_pred"]),
                 "MAE": float(np.mean(np.abs(g["y_true"]-g["y_pred"]))),
                 "n": int(len(g))
             }))
             .reset_index()
          )
by_item.sort_values("WAPE").to_csv(MET_DIR/"val_by_item_metrics.csv", index=False, encoding="utf-8-sig")

# 상/하위 품목 막대그래프 (WAPE)
plt.figure()
plot_df = by_item.sort_values("WAPE").head(10)
plt.bar(plot_df["item"], plot_df["WAPE"])
plt.xticks(rotation=60)
plt.title("검증 WAPE 상위 10개(낮을수록 좋음)")
plt.tight_layout()
plt.savefig(FIG_DIR/"val_wape_top10.png", dpi=150); plt.close()

plt.figure()
plot_df = by_item.sort_values("WAPE").tail(10)
plt.bar(plot_df["item"], plot_df["WAPE"])
plt.xticks(rotation=60)
plt.title("검증 WAPE 하위 10개(개선 대상)")
plt.tight_layout()
plt.savefig(FIG_DIR/"val_wape_bottom10.png", dpi=150); plt.close()

# 큰 오차 사례 Top-N (진단)
val["abs_err"] = (val["y_true"]-val["y_pred"]).abs()
worst = val.sort_values("abs_err", ascending=False).head(50)
worst.to_csv(MET_DIR/"val_worst_cases_top50.csv", index=False, encoding="utf-8-sig")

# 대표 품목 6개 예측 vs 실측 라인
show_items = by_item.sort_values("WAPE").head(3)["item"].tolist() + by_item.sort_values("WAPE").tail(3)["item"].tolist()
for it in show_items:
    # 실측을 붙이기 위해 weekly_agg에서 y_true 시점(=week+1주)의 실제 단가를 구해도 되지만
    # val 파일에 이미 y_true가 있으므로 그것으로 사용
    sub = val[val["item"]==it].sort_values("week")
    plt.figure()
    plt.plot(sub["week"], sub["y_true"], label="실측(다음주)")
    plt.plot(sub["week"], sub["y_pred"], label="예측")
    plt.title(f"[검증] {it} - 다음주 단가 예측 vs 실측")
    plt.xlabel("week"); plt.ylabel("단가")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR/f"val_line_{it}.png", dpi=150); plt.close()

# 5) 다음 주 예측: 전주 대비 변동률 / 리스크 랭킹
nextp = pd.read_csv(NEXT_PRED_CSV, parse_dates=["week"])
# 비교용으로 '현재 주' 실측 단가 필요 → weekly_agg에서 해당 주 단가 붙임
last_week = nextp["week"].max()
wk_last = wk[wk["week"]==last_week][["item","unit_price_week"]].rename(columns={"unit_price_week":"this_week_price"})
df_next = nextp.merge(wk_last, on="item", how="left")
df_next["pct_change_vs_this_week"] = (df_next["pred_next_week_unit_price"] - df_next["this_week_price"]) / (df_next["this_week_price"]+1e-9)

# 급등/급락 Top-N
top_up   = df_next.sort_values("pct_change_vs_this_week", ascending=False).head(15)
top_down = df_next.sort_values("pct_change_vs_this_week", ascending=True).head(15)
top_up.to_csv(MET_DIR/"nextweek_top_increase.csv", index=False, encoding="utf-8-sig")
top_down.to_csv(MET_DIR/"nextweek_top_decrease.csv", index=False, encoding="utf-8-sig")

plt.figure(); plt.bar(top_up["item"], top_up["pct_change_vs_this_week"]); plt.xticks(rotation=60)
plt.title("다음 주 급등 예상 Top-15"); plt.tight_layout()
plt.savefig(FIG_DIR/"nextweek_top_increase.png", dpi=150); plt.close()

plt.figure(); plt.bar(top_down["item"], top_down["pct_change_vs_this_week"]); plt.xticks(rotation=60)
plt.title("다음 주 급락 예상 Top-15"); plt.tight_layout()
plt.savefig(FIG_DIR/"nextweek_top_decrease.png", dpi=150); plt.close()

print("[완료] 보고서 파일 생성:")
for p in [MET_DIR/"master_raw_item_summary.csv",
          MET_DIR/"weekly_agg_summary.csv",
          MET_DIR/"feature_null_rate_top30.csv",
          MET_DIR/"val_overall_metrics.csv",
          MET_DIR/"val_by_item_metrics.csv",
          MET_DIR/"val_worst_cases_top50.csv",
          MET_DIR/"nextweek_top_increase.csv",
          MET_DIR/"nextweek_top_decrease.csv"]:
    print(" -", p)
