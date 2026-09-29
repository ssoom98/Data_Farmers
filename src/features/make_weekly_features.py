# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from src.config import WEEKLY_AGG_PQ, WEATHER_CSV, WEEKLY_FEAT_PQ, ensure_dirs

def load_weather_weekly() -> pd.DataFrame:
    # 유연한 컬럼 매핑
    for enc in ["utf-8-sig","utf-8","cp949"]:
        try:
            wdf = pd.read_csv(WEATHER_CSV, encoding=enc)
            break
        except Exception:
            wdf = None
    if wdf is None:
        raise RuntimeError(f"기상 CSV 읽기 실패: {WEATHER_CSV}")
    if "조회일자" in wdf.columns:
        wdf["date"] = pd.to_datetime(wdf["조회일자"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in wdf.columns:
        wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
    else:
        raise ValueError("기상 CSV에 날짜 컬럼(조회일자/date)이 필요합니다.")

    cmap = {
        "평균기온(°C)":"tavg","최저기온(°C)":"tmin","최고기온(°C)":"tmax",
        "강수량(mm)":"precip","평균상대습도(%)":"humidity",
        "일조시간(hr)":"sunshine","일사량(MJ/m2)":"radiation",
        "평균풍속(m/s)":"wind",
        "tavg":"tavg","tmin":"tmin","tmax":"tmax","precip":"precip",
        "humidity":"humidity","sunshine":"sunshine","radiation":"radiation","wind":"wind"
    }
    w = pd.DataFrame({"date": wdf["date"]})
    for src, dst in cmap.items():
        if src in wdf.columns:
            w[dst] = pd.to_numeric(wdf[src], errors="coerce")

    if len([c for c in w.columns if c!="date"]) == 0:
        # 기상 변수 없으면 주차 인덱스만
        w_week = w.set_index("date").resample("W-MON").size().reset_index().rename(columns={"date":"week"}).drop(columns=0)
    else:
        agg = {c:("sum" if c in ["precip","sunshine","radiation"] else "mean") for c in w.columns if c!="date"}
        w_week = w.set_index("date").resample("W-MON").agg(agg).reset_index().rename(columns={"date":"week"})
    return w_week

def add_lag_roll(g: pd.DataFrame, col: str) -> pd.DataFrame:
    g = g.sort_values("week").copy()
    for L in [1,2,4,8,12,52]:
        g[f"{col}_lag{L}"] = g[col].shift(L)
    for W in [4,8,12,26,52]:
        g[f"{col}_rollmean{W}"] = g[col].shift(1).rolling(W, min_periods=max(2,int(W/2))).mean()
        g[f"{col}_rollstd{W}"]  = g[col].shift(1).rolling(W, min_periods=max(2,int(W/2))).std()
    g[f"{col}_yoy_gap"] = g[col] - g[f"{col}_lag52"]
    return g

def main():
    ensure_dirs()
    g = pd.read_parquet(WEEKLY_AGG_PQ)
    w_week = load_weather_weekly()
    feat = g.merge(w_week, on="week", how="left")

    feat["log_price"] = np.log1p(feat["unit_price_week"])
    feat["log_qty"]   = np.log1p(feat["qty_in_week"])

    feat = feat.groupby("item", group_keys=False).apply(add_lag_roll, col="log_price")
    feat = feat.groupby("item", group_keys=False).apply(add_lag_roll, col="log_qty")

    # 변동성
    def add_vol(df):
        df = df.sort_values("week").copy()
        for W in [4,8,12]:
            df[f"price_vol_{W}"] = df["unit_price_week"].pct_change().shift(1).rolling(W, min_periods=2).std()
        return df
    feat = feat.groupby("item", group_keys=False).apply(add_vol)

    # 캘린더
    feat["year"] = pd.to_datetime(feat["week"]).dt.year
    feat["month"] = pd.to_datetime(feat["week"]).dt.month
    woy = pd.to_datetime(feat["week"]).dt.isocalendar().week.astype(int)
    feat["weekofyear"] = woy
    # 타깃(다음 주)
    feat["next_week_unit_price"] = feat.groupby("item")["unit_price_week"].shift(-1)
    feat["next_week_log_price"]  = feat.groupby("item")["log_price"].shift(-1)

    feat.to_parquet(WEEKLY_FEAT_PQ, index=False)
    print(f"[OK] weekly_features.parquet 저장 → {WEEKLY_FEAT_PQ}")

if __name__ == "__main__":
    main()
