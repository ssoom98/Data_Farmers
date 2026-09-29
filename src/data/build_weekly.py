# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from src.config import MASTER_RAW_CSV, WEEKLY_AGG_PQ, ensure_dirs

def main():
    ensure_dirs()
    df = pd.read_csv(MASTER_RAW_CSV, parse_dates=["date"])
    df["week"] = df["date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    g = df.groupby(["item","week"], as_index=False).agg(
        qty_in_week=("qty_in","sum"),
        amount_week=("amount","sum")
    )
    # 견고한 주간 단가
    valid = (g["qty_in_week"]>0) & (g["amount_week"]>0)
    g["unit_price_week"] = np.where(valid, g["amount_week"]/g["qty_in_week"], np.nan)
    g.to_parquet(WEEKLY_AGG_PQ, index=False)
    print(f"[OK] weekly_agg.parquet 저장 → {WEEKLY_AGG_PQ}")

if __name__ == "__main__":
    main()
