# src/eda/quick_checks.py
import pandas as pd, numpy as np
from pathlib import Path

MASTER = Path("data/processed/master_raw.csv")
WEEKLY = Path("data/processed/weekly_agg.parquet")
OUT_FIG = Path("reports/figures"); OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_MET = Path("reports/metrics"); OUT_MET.mkdir(parents=True, exist_ok=True)

def run():
    dfm = pd.read_csv(MASTER, parse_dates=["date"])
    wk = pd.read_parquet(WEEKLY)
    # 기본 품질
    q = {
      "master_rows": len(dfm),
      "master_null_rate": dfm.isna().mean().to_dict(),
      "weekly_rows": len(wk),
      "weekly_null_rate": wk.isna().mean().to_dict(),
      "unit_price_nonpos": int((wk["unit_price_week"]<=0).sum())
    }
    pd.Series(q["master_null_rate"]).to_csv(OUT_MET/"null_master.csv")
    pd.Series(q["weekly_null_rate"]).to_csv(OUT_MET/"null_weekly.csv")
    (wk.groupby("item")["unit_price_week"]
      .describe()[["count","mean","std","min","25%","50%","75%","max"]]
    ).to_csv(OUT_MET/"desc_weekly_price_by_item.csv", encoding="utf-8-sig")
    print("[EDA] 저장 완료:", OUT_MET)

if __name__ == "__main__":
    run()
