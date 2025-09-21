# -*- coding: utf-8 -*-
"""
Full pipeline:
1) For each product {key}:
   - load '{key}_가공.csv' from interim folder
   - merge with weather_cut (processed weather) using '거래일자'
   - train weather models (LightGBM) on historical data, then iterative-forecast future weather for horizon
   - create weather-based lag/rolling/ewma features from predicted+historical weather
   - train price models:
       * LightGBM (many features)
       * SARIMAX (limited exog list per spec)
   - forecast prices for horizon
   - ensemble (0.6 LGBM + 0.4 SARIMAX)
2) Collect daily predicted prices for all products; per day, per cluster choose product with lowest predicted price.
3) Save outputs to CSVs.
"""

import os
import copy
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# modeling
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ========== User settings ==========
INTERIM_DIR = r"C:\price_prediction\scaffold\data\interim"
PROCESSED_DIR = r"C:\price_prediction\scaffold\data\processed"
WEATHER_CUT_PATH = os.path.join(PROCESSED_DIR, "weather_cut.csv")  # merged weather base file
OUTPUT_DIR = r"C:\price_prediction\scaffold\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prediction horizon
START_DATE = pd.to_datetime("2025-09-21")  # inclusive
END_DATE   = pd.to_datetime("2025-12-31")  # inclusive
FUTURE_DATES = pd.date_range(START_DATE, END_DATE, freq="D")

# product->cluster dictionary (from user's message)
CLUSTER_DICT = {
 '깐양파':2, '깐쪽파':0, '깻잎(일반)':4, '노랑파프리카':3, '녹광':3, '뉴그린':4,
 '느타리버섯(일반)':1, '단호박':6, '당근(일반)':2, '대추방울':5, '대파(일반)':2,
 '돌미나리':4, '레드치커리':4, '만생양파':2, '맛느타리버섯':1, '미나리(일반)':4,
 '밤고구마':6, '방울토마토':5, '백다다기':4, '브로코리(국산)':4, '빨강파프리카':3,
 '새송이(일반)':1, '수미':6, '시금치(일반)':4, '애호박':4, '양배추(일반)':5,
 '양상추(일반)':4, '양송이(일반)':1, '영양부추':4, '오이맛고추':3, '완숙토마토':5,
 '일반부추':4, '자주양파':2, '적채(일반)':5, '쥬키니호박':6, '쪽파(일반)':0,
 '청양':3, '청초(일반)':3, '청피망':3, '취청':4, '치커리(일반)':4, '치콘':4,
 '토마토(일반)':5, '팽이':1, '포장쪽파':0, '표고버섯(일반)':1, '호박고구마':6,
 '홍고추(일반)':3, '홍청양':3, '홍피망':3
}

PRODUCT_KEYS = list(CLUSTER_DICT.keys())

# Weather targets (as specified)
WEATHER_TARGETS = [
    '평균 기온(°C)', '최고 기온(°C)', '최저 기온(°C)',
    '평균 강수량(mm)', '평균 일사량(MJ/㎡)'
]

# SARIMAX exog columns required by user
SARIMAX_EXOG_LIST = [
    'lag_avtemp_30','lag_maxtemp_30','lag_mintemp_30','lag_rain_30','lag_light_30',
    'rolling_mean_avtemp_90','rolling_mean_rain_90','rolling_mean_light_90',
    'rolling_sum_rain_90','rolling_sum_rain_180','rolling_sum_light_90','rolling_sum_light_180'
    # plus: '평균 기온...', '최고 기온...', etc. (we'll append those later)
]

# Ensemble weights
W_LGBM = 0.6
W_SARIMAX = 0.4

# LightGBM params (simple default)
LGB_PARAMS = {
    'objective': 'regression',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# =========================================
# Utility functions
# =========================================

def load_and_merge_product(product_key):
    """
    Load {key}_가공.csv from interim and merge with weather_cut.csv on '거래일자'.
    Returns merged pandas DataFrame with datetime index (거래일자).
    """
    prod_path = os.path.join(INTERIM_DIR, f"{product_key}_가공.csv")
    if not os.path.exists(prod_path):
        raise FileNotFoundError(f"{prod_path} not found.")
    
    # 여러 인코딩 시도하여 파일 읽기
    encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
    df_prod = None
    
    for encoding in encodings_to_try:
        try:
            df_prod = pd.read_csv(prod_path, parse_dates=['거래일자'], encoding=encoding)
            print(f"Product file {product_key} loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df_prod is None:
        raise Exception(f"Could not read {prod_path} with any of the supported encodings")
    
    # load weather_cut
    if not os.path.exists(WEATHER_CUT_PATH):
        raise FileNotFoundError(f"{WEATHER_CUT_PATH} not found.")
    
    # weather 파일도 여러 인코딩 시도
    df_weather_base = None
    for encoding in encodings_to_try:
        try:
            df_weather_base = pd.read_csv(WEATHER_CUT_PATH, parse_dates=['거래일자'], encoding=encoding)
            print(f"Weather file loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df_weather_base is None:
        raise Exception(f"Could not read {WEATHER_CUT_PATH} with any of the supported encodings")
    
    # merge on 거래일자 (left join product)
    df = pd.merge(df_prod, df_weather_base, on='거래일자', how='left', suffixes=('','_w'))
    df = df.sort_values('거래일자').reset_index(drop=True)
    df['거래일자'] = pd.to_datetime(df['거래일자'])
    df = df.set_index('거래일자')
    return df

def select_weather_features_for_model(df):
    """
    Select X features for weather model per user's rule:
      - columns starting with 'lag' but not containing 'trading'
      - columns starting with 'rolling'
      - columns starting with 'ewma'
      - '월', '주차', '공휴일', columns starting with '요일_'
    """
    cols = []
    for c in df.columns:
        low = c.lower()
        if c.startswith('lag') and ('trading' not in c):
            cols.append(c)
        elif c.startswith('rolling') or c.startswith('ewma'):
            cols.append(c)
        elif c in ['월','주차','공휴일']:
            cols.append(c)
        elif c.startswith('요일_'):
            cols.append(c)
    return [c for c in cols if c in df.columns]

def train_weather_models(df_hist, features_X, targets=WEATHER_TARGETS):
    """
    Train LightGBM models for each weather target using historical rows up to yesterday.
    Returns dict of fitted models.
    """
    models = {}
    # Drop rows with all-nans in targets
    train_df = df_hist.copy().dropna(subset=targets, how='all')
    # for simplicity, train one regressor per target
    X = train_df[features_X].fillna(method='ffill').fillna(0)
    for tgt in targets:
        if tgt not in train_df.columns:
            raise KeyError(f"Target {tgt} not found in dataframe")
        y = train_df[tgt].fillna(method='ffill').fillna(0)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        # LightGBM 최신 버전 호환성을 위해 callbacks 사용
        try:
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        except:
            # 구버전 호환성을 위한 fallback
            m.fit(X_tr, y_tr)
        models[tgt] = m
    return models

def iterative_forecast_weather(df_hist, models, features_X, future_dates):
    """
    Given trained weather models and historical df (index datetime), produce predictions for the future_dates.
    Uses iterative forecasting: for each future day t, compute feature vector using history + previously predicted days.
    Returns a DataFrame future_df indexed by future_dates with columns WEATHER_TARGETS.
    """
    # We will work on a growing df that contains historical weather (with targets) and then append predicted rows
    # Copy the historical df for temp storage
    df_work = df_hist.copy()
    # For safety, ensure features exist (fill missing)
    for c in features_X:
        if c not in df_work.columns:
            df_work[c] = 0.0
    # Prepare storage
    fut_records = []
    for d in future_dates:
        # construct X_row using selection function: features may be lag/rolling computed from df_work (which includes prior preds)
        # We recompute rolling/lag/ewma features dynamically if they are defined as columns in df_hist. For simplicity,
        # if the feature exists and is up-to-date in df_work, use last row; if not, try to compute common patterns.
        # Easiest robust approach: assume user-provided lag/rolling/ewma columns are precomputed in historical df.
        X_row = {}
        last_index = df_work.index.max()
        # We must set the calendar features: 월, 주차, 공휴일, 요일_
        X_row['월'] = d.month
        X_row['주차'] = int(d.isocalendar()[1])
        # public holiday flag: if present in hist as column '공휴일', try to map by date; else assume 0
        if '공휴일' in df_hist.columns:
            # try to get exact value from df_hist if available by date equality
            val = 0
            if d in df_hist.index:
                val = int(df_hist.at[d, '공휴일']) if not pd.isna(df_hist.at[d, '공휴일']) else 0
            X_row['공휴일'] = val
        else:
            X_row['공휴일'] = 0
        # 요일_ dummy columns
        for i in range(7):
            col = f'요일_{["월","화","수","목","금","토","일"][i]}'
            X_row[col] = 1 if d.weekday() == i else 0

        # For every other feature requested in features_X, if it's a lag/rolling/ewma that exists in df_hist columns,
        # we will try to compute its value for date d by applying pandas rolling/shift on df_work's relevant base series.
        # To keep the implementation robust without parsing every custom column name, we attempt:
        # - If column exists in df_work index last row, use the last value (this handles many precomputed static features)
        # - Otherwise fill with 0 (safe fallback)
        for c in features_X:
            if c in ['월','주차','공휴일'] or c.startswith('요일_'):
                continue
            # if feature already exists in df_work (i.e., its latest value corresponds to last history date),
            # we attempt to create the same feature for the new date by shifting/rolling on the base series if possible.
            # Here we implement a conservative approach: if col like 'lag_*' or 'rolling_*' exists historically, we recompute using base columns if possible.
            if c in df_work.columns:
                # If c is a lag like 'lag_avtemp_30' we attempt to parse base series and window
                if c.startswith('lag_'):
                    # common naming patterns in user's spec:
                    # lag_avtemp_30, lag_maxtemp_60 etc.
                    parts = c.split('_')
                    # pattern: 'lag', base, window
                    if len(parts) >= 3:
                        base = parts[1]  # e.g., 'avtemp','maxtemp','rain','light','mintemp'
                        window = int(parts[-1])
                        # map base name to actual weather column if possible
                        mapping = {
                            'avtemp':'평균 기온(°C)',
                            'maxtemp':'최고 기온(°C)',
                            'mintemp':'최저 기온(°C)',
                            'rain':'평균 강수량(mm)',
                            'light':'평균 일사량(MJ/㎡)'
                        }
                        base_col = mapping.get(base, None)
                        if base_col and base_col in df_work.columns:
                            # compute value at date d-window (i.e., lag window)
                            target_date = d - pd.Timedelta(days=window)
                            if target_date in df_work.index:
                                X_row[c] = df_work.at[target_date, base_col]
                            else:
                                # if target date not available, fallback to last available
                                X_row[c] = df_work[base_col].iloc[-1]
                        else:
                            X_row[c] = df_work[c].iloc[-1] if c in df_work.columns else 0.0
                    else:
                        X_row[c] = df_work[c].iloc[-1] if c in df_work.columns else 0.0
                elif c.startswith('rolling_') or c.startswith('ewma_'):
                    # If feature exists historically, we attempt to recompute using base series (best-effort)
                    # fallback: use last available value
                    X_row[c] = df_work[c].iloc[-1] if c in df_work.columns else 0.0
                else:
                    # generic fallback: use last value if present
                    X_row[c] = df_work[c].iloc[-1] if c in df_work.columns else 0.0
            else:
                # not in df_work, default 0
                X_row[c] = 0.0

        # Form DataFrame row
        X_row_df = pd.DataFrame([X_row], index=[d])
        # Ensure columns order to match training features (missing -> fill 0)
        X_row_df = X_row_df.reindex(columns=features_X, fill_value=0.0)

        # Predict each weather target using models
        preds = {}
        for tgt, model in models.items():
            try:
                preds[tgt] = float(model.predict(X_row_df)[0])
            except Exception:
                # fallback: use last observed value
                preds[tgt] = float(df_work[tgt].iloc[-1]) if tgt in df_work.columns else 0.0

        # Append predicted weather to df_work as a new row so that future lags/rollings can use it
        new_row = X_row.copy()
        for tgt in WEATHER_TARGETS:
            new_row[tgt] = preds.get(tgt, 0.0)
        # Add base columns if not exist in df_work
        missing_cols = set(new_row.keys()) - set(df_work.columns)
        for c in missing_cols:
            df_work[c] = np.nan
        df_work.loc[d] = pd.Series(new_row)
        fut_records.append(pd.Series(new_row)[WEATHER_TARGETS])

    future_df = pd.DataFrame(fut_records, index=future_dates)[WEATHER_TARGETS]
    return future_df

# helper function to build derived weather features for entire series df (history+future)
def create_weather_derived_features(df_all):
    """
    Given df_all indexed by date and containing weather base columns,
    produce the derived features described in user's spec and return updated df_all.
    The function computes:
      - lags for the 5 weather vars: 30,60,90,180 -> names: lag_avtemp_30 etc.
      - rolling_mean for windows 30,90,180 (shifted by 1): rolling_mean_avtemp_30 etc.
      - rolling_sum for rain and light windows 30,90,180 (shifted by 1)
      - rolling_std for windows 30 for all five vars (shifted 1)
      - ewma spans 14,60 for all five vars (shifted 1)
    """
    df = df_all.copy()
    # mapping between short base names and actual column names
    map_base = {
        'avtemp':'평균 기온(°C)',
        'maxtemp':'최고 기온(°C)',
        'mintemp':'최저 기온(°C)',
        'rain':'평균 강수량(mm)',
        'light':'평균 일사량(MJ/㎡)'
    }

    # 1) LAGS: 30,60,90,180
    for short, col in map_base.items():
        for lag_day in [30,60,90,180]:
            newcol = f'lag_{short}_{lag_day}'
            df[newcol] = df[col].shift(lag_day)

    # 2) Rolling means (exclude today -> shift(1))
    for short, col in map_base.items():
        for win in [30,90,180]:
            newcol = f'rolling_mean_{short}_{win}'
            df[newcol] = df[col].shift(1).rolling(window=win, min_periods=1).mean()

    # 3) Rolling sums for rain & light
    for short, col in [('rain','평균 강수량(mm)'), ('light','평균 일사량(MJ/㎡)')]:
        for win in [30,90,180]:
            newcol = f'rolling_sum_{short}_{win}'
            df[newcol] = df[col].shift(1).rolling(window=win, min_periods=1).sum()

    # 4) Rolling std 30 for each var (exclude today)
    for short, col in map_base.items():
        newcol = f'rolling_std_{short}_30'
        df[newcol] = df[col].shift(1).rolling(window=30, min_periods=1).std()

    # 5) EWMA spans 14,60 (exclude today -> shift(1))
    for short, col in map_base.items():
        for span in [14,60]:
            newcol = f'ewma_{short}_span{span}'
            df[newcol] = df[col].shift(1).ewm(span=span, adjust=False).mean()

    # fill NaNs conservatively
    df = df.fillna(method='ffill').fillna(0)
    return df

# function to prepare LGBM price features per user's rules
def prepare_price_X_y(df_full):
    """
    Per user's rules for LGBM X features:
      - include predicted base weather vars (the five targets)
      - include all lag* columns that do NOT contain 'trading' AND also include any columns with 'trading'
      - include all rolling* columns
      - include columns starting with '요일_'
      - include '공휴일'
      - include existing trading columns (those containing 'trading')
    y = '가격'
    """
    cols = []
    # 1) add five weather base columns if present
    for c in WEATHER_TARGETS:
        if c in df_full.columns:
            cols.append(c)

    # 2) lag* excluding contain 'trading'
    for c in df_full.columns:
        if c.startswith('lag') and ('trading' not in c):
            cols.append(c)
    # 3) rolling*
    for c in df_full.columns:
        if c.startswith('rolling'):
            cols.append(c)
    # 4) ewma*
    for c in df_full.columns:
        if c.startswith('ewma'):
            cols.append(c)
    # 5) trading columns (columns that include 'trading' anywhere)
    for c in df_full.columns:
        if 'trading' in c:
            cols.append(c)
    # 6) weekday dummies and holiday
    for c in df_full.columns:
        if c.startswith('요일_') or c=='공휴일':
            cols.append(c)
    # deduplicate while preserving order
    seen = set()
    price_X_cols = [x for x in cols if not (x in seen or seen.add(x))]
    # target
    y_col = '가격'
    return df_full[price_X_cols].fillna(0), df_full[y_col].fillna(0)

# function to prepare SARIMAX exog & y per user's rules
def prepare_sarimax_exog_y(df_full):
    """
    SARIMAX exog should include:
      - lag_avtemp_30, lag_maxtemp_30, lag_mintemp_30, lag_rain_30, lag_light_30
      - rolling_mean_avtemp_90, rolling_mean_rain_90, rolling_mean_light_90
      - rolling_sum_rain_90, rolling_sum_rain_180, rolling_sum_light_90, rolling_sum_light_180
      - plus base weather vars: the five targets
      - plus '요일_' columns, '공휴일', and all 'trading' columns
    """
    required = copy.copy(SARIMAX_EXOG_LIST)
    # add weather base columns
    for c in WEATHER_TARGETS:
        required.append(c)
    # add weekdays, holiday, trading columns
    for c in df_full.columns:
        if c.startswith('요일_') or c=='공휴일' or ('trading' in c):
            required.append(c)
    # select available
    exog_cols = [c for c in required if c in df_full.columns]
    # y
    y = df_full['가격'].fillna(0)
    X_exog = df_full[exog_cols].fillna(0)
    return X_exog, y

# main per-product pipeline
def process_product(product_key):
    print(f"\n=== Processing product: {product_key} ===")
    # 1) load & merge
    df = load_and_merge_product(product_key)  # index is 거래일자
    # ensure the five weather base columns exist in df (from weather_cut)
    for c in WEATHER_TARGETS:
        if c not in df.columns:
            df[c] = np.nan

    # ensure price column exists
    if '가격' not in df.columns:
        raise KeyError(f"'가격' column missing in product file {product_key}")

    # Split historical portion: all dates < START_DATE considered history
    hist_df = df[df.index < START_DATE].copy()
    if hist_df.empty:
        raise ValueError(f"No historical rows prior to {START_DATE} for product {product_key}")

    # 2) Weather model training
    weather_features = select_weather_features_for_model(df)
    if len(weather_features) == 0:
        # fallback: use calendar features
        weather_features = ['월','주차','공휴일'] + [c for c in df.columns if c.startswith('요일_')]
    print(f"Weather features used ({len(weather_features)}): {weather_features[:10]}{'...' if len(weather_features)>10 else ''}")

    weather_models = train_weather_models(hist_df, weather_features, targets=WEATHER_TARGETS)

    # 3) iterative forecast weather for FUTURE_DATES
    future_weather_df = iterative_forecast_weather(hist_df, weather_models, weather_features, FUTURE_DATES)
    # persist predicted weather for later joining (optional) - utf-8로 저장
    future_weather_df.to_csv(os.path.join(OUTPUT_DIR, f"{product_key}_predicted_weather.csv"), encoding='utf-8-sig')

    # 4) combine historical + predicted weather into df_all for feature generation
    # Build df_all with index from hist_df.index union FUTURE_DATES
    df_all = pd.concat([hist_df, future_weather_df], axis=0, sort=False)
    # Ensure weather base columns exist across
    for c in WEATHER_TARGETS:
        if c not in df_all.columns:
            df_all[c] = np.nan
    # If predicted future rows don't have other columns, fill from last known
    df_all = df_all.sort_index()
    df_all = df_all.fillna(method='ffill').fillna(0)

    # 5) create weather-derived features over df_all
    df_all = create_weather_derived_features(df_all)

    # 6) Build final dataframe for price modeling: we'll restrict to index up to END_DATE
    df_model = df_all.copy()
    # Prepare LGBM features & label
    X_price_all, y_price_all = prepare_price_X_y(df_model)
    # split historical rows for training LGBM: use rows with index < START_DATE and with non-null y
    train_idx = df_model.index < START_DATE
    X_train_price = X_price_all[train_idx]
    y_train_price = y_price_all[train_idx]
    # For validation, keep a small tail of historical for early stopping
    if len(X_train_price) < 30:
        print("Warning: small training data for price model, results may be unstable.")
    # train LightGBM price model
    lgb_price = lgb.LGBMRegressor(**LGB_PARAMS)
    # simple split: last 20% of training as val
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_price, y_train_price, test_size=0.2, shuffle=False)
        # LightGBM 최신 버전 호환성을 위해 callbacks 사용
        try:
            lgb_price.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        except:
            # 구버전 호환성을 위한 fallback
            lgb_price.fit(X_tr, y_tr)
    except Exception:
        # fallback: fit on all
        lgb_price.fit(X_train_price, y_train_price)

    # 7) Prepare SARIMAX exog & y for training (historical only)
    X_exog_all, y_all = prepare_sarimax_exog_y(df_model)
    X_exog_train = X_exog_all[df_model.index < START_DATE]
    y_train_sarimax = y_all[df_model.index < START_DATE]
    # Fit SARIMAX - simple order; user can tune
    try:
        sarimax_mod = SARIMAX(y_train_sarimax, exog=X_exog_train, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
        sarimax_res = sarimax_mod.fit(disp=False, maxiter=50)
    except Exception as e:
        print("SARIMAX failed to converge or fit; falling back to naive SARIMAX with no exog.")
        sarimax_mod = SARIMAX(y_train_sarimax, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
        sarimax_res = sarimax_mod.fit(disp=False, maxiter=50)

    # 8) Forecast prices for FUTURE_DATES
    # For LGBM: prepare future X (take df_model rows for FUTURE_DATES)
    X_future_price = X_price_all.reindex(FUTURE_DATES).fillna(method='ffill').fillna(0)
    pred_lgbm = lgb_price.predict(X_future_price)

    # For SARIMAX: exog for future
    X_exog_future = X_exog_all.reindex(FUTURE_DATES).fillna(method='ffill').fillna(0)
    try:
        pred_sarimax = sarimax_res.get_forecast(steps=len(FUTURE_DATES), exog=X_exog_future).predicted_mean
        pred_sarimax = np.asarray(pred_sarimax)
    except Exception:
        # fallback: use last observed mean
        last_val = y_train_sarimax.iloc[-1] if len(y_train_sarimax)>0 else 0.0
        pred_sarimax = np.array([last_val]*len(FUTURE_DATES))

    # 9) Ensemble
    pred_ensemble = W_LGBM * pred_lgbm + W_SARIMAX * pred_sarimax

    # 10) Build output DataFrame per product
    out_df = pd.DataFrame({
        '거래일자': FUTURE_DATES,
        '가격_pred_lgbm': pred_lgbm,
        '가격_pred_sarimax': pred_sarimax,
        '가격_pred_ensemble': pred_ensemble
    }).set_index('거래일자')

    # utf-8 인코딩으로 저장
    out_df.to_csv(os.path.join(OUTPUT_DIR, f"{product_key}_price_forecast.csv"), encoding='utf-8-sig')

    return out_df[['가격_pred_ensemble']]

# ===========================
# Run pipeline for all products
# ===========================
all_preds = {}  # dict product -> series of ensemble preds (indexed by FUTURE_DATES)

for key in tqdm(PRODUCT_KEYS):
    try:
        preds = process_product(key)  # returns DataFrame with 가격_pred_ensemble
        all_preds[key] = preds['가격_pred_ensemble']
    except Exception as e:
        print(f"Error processing {key}: {e}")
        # create NaN series for consistency
        all_preds[key] = pd.Series([np.nan]*len(FUTURE_DATES), index=FUTURE_DATES)

# Combine all predictions into one DataFrame: columns are products
preds_df = pd.DataFrame(all_preds, index=FUTURE_DATES)
preds_df.index.name = '거래일자'
# utf-8 인코딩으로 저장
preds_df.to_csv(os.path.join(OUTPUT_DIR, "all_products_price_forecast.csv"), encoding='utf-8-sig')

# ===========================
# For each date, find lowest-price product per cluster
# ===========================
# Build cluster mapping: cluster -> list of products
cluster_map = {}
for prod, cluster in CLUSTER_DICT.items():
    cluster_map.setdefault(cluster, []).append(prod)

# Results DataFrame: index FUTURE_DATES, columns cluster_<k> giving product with min price and price
cluster_results = []
for d in FUTURE_DATES:
    row = {'거래일자': d}
    for cluster_id, products in cluster_map.items():
        # subset predictions for that date and products
        day_prices = preds_df.loc[d, products]
        # handle all-NaN
        if day_prices.isna().all():
            row[f'cluster_{cluster_id}_product'] = None
            row[f'cluster_{cluster_id}_price'] = np.nan
        else:
            # choose product with minimum predicted price; if ties, first
            min_prod = day_prices.idxmin()
            min_price = float(day_prices[min_prod])
            row[f'cluster_{cluster_id}_product'] = min_prod
            row[f'cluster_{cluster_id}_price'] = min_price
    cluster_results.append(row)

cluster_out_df = pd.DataFrame(cluster_results).set_index('거래일자')
# utf-8 인코딩으로 저장
cluster_out_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_daily_min_products.csv"), encoding='utf-8-sig')

print("Pipeline finished. Outputs saved to:", OUTPUT_DIR)

