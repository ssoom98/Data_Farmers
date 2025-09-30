
---

# 급식용 농산물 가격 예측 및 대체 식재료 추천 시스템

급식 발주·예산 의사결정 단위(주차)에 맞춘 **주간 단가 시계열**을 구축하고, **기상·수급·계절성**을 반영한 피처엔지니어링과 **Global GBM(LightGBM)** 회귀 모델로 다음 주 단가를 예측합니다. 예측 결과를 바탕으로 **동일 품목군 내 대체재 추천**을 생성하여 **원가 절감과 공급 안정성**을 지원합니다.

---

## 1. 데이터 전처리 및 데이터 탐색

### 1-1) 수집/통합 단계 (`src/data/build_master_raw.py`)

**목적**
여러 원천 CSV를 **표준 스키마**로 일관 통합하여 `data/processed/master_raw.csv`를 생성합니다. 인코딩(UTF-8/CP949) 자동 시도와 컬럼 자동 매핑으로 **재현 가능한 입력 기준선**을 확립합니다.

**표준 스키마**

* `date`, `item`, `qty_in`, `amount`

**자동 매핑 규칙**

* `date`: `["거래일자","date"]`
* `item`: `["품목명","품목","item"]`
* `qty_in`: `["반입량","수량","qty_in"]`
* `amount`: `["금액","amount"]`

> 필수 컬럼 누락 시 처리 중단(`ValueError`)으로 **조기 오류 감지**.

---

### 1-2) 주간 집계 및 기본 파생 (`src/data/build_weekly.py`)

**목적**
발주·예산의 실무 단위인 **주(Week)**로 정규화하여 잡음을 줄이고, 이후 **다음 주 단가 예측**과 계절성 분석을 위한 안정적 테이블을 만듭니다.

**핵심 로직**

1. **주차 기준**: `week = date.to_period("W-MON").start_time` (월요일 시작으로 고정)
2. **집계(그룹: `["item","week"]`)**

   * `qty_in_week = Σ qty_in`, `amount_week = Σ amount`
3. **견고한 주간 단가**

   * 유효 조건: `qty_in_week > 0` and `amount_week > 0`
   * `unit_price_week = amount_week / qty_in_week` (그 외 `NaN`)

**산출 스키마**: `data/processed/weekly_agg.parquet` (행: 품목×주)

* `item`(str), `week`(ts; 월요일 00:00), `qty_in_week`(float), `amount_week`(float), `unit_price_week`(float, nullable)

**왜 주간 단가인가?**

* **실무 적합성**: 급식 발주/예산이 주 단위로 움직임
* **잡음 감소**: 휴장·이벤트·특이치 효과 완화 → **추세/계절성 신호 강화**
* **모델 안정성**: 시계열/회귀 모두에서 분산 감소

**데이터 품질 포인트**

* 0 또는 음수 합계 방지, 결측 단가 `NaN` 관리 정책(유지/제한적 대치), 연말·연초 경계 주차 점검, 품목별 **연속 주차 수** 확보

---

### 1-3) EDA/검증 시각화 (실제 산출 이미지)

* **주간 커버리지 히트맵**(품목×주, 거래 유무 0/1)

<img width="5637" height="1295" alt="coverage_heatmap_top40" src="https://github.com/user-attachments/assets/a0b4a403-fcc5-4b96-aa9f-b8d74b0dc29b" />
  

* **단가 분포(박스플롯)**(품목별 `unit_price_week` 분포 및 이상치 후보)

<img width="2700" height="1080" alt="price_boxplot_top30" src="https://github.com/user-attachments/assets/3fac8fbe-f680-41a9-b2a6-8a3e59673eef" />


* **거래량-단가 산점도**(수급 효과: `qty_in_week ↑ → unit_price_week ↓`)

 <div style="display: flex; justify-content: center; gap: 10px;">

  <img src="https://github.com/user-attachments/assets/524f30a0-19a9-4e3b-a40f-70fbcb45eecf" alt="qty_vs_price_scatter_all" width="250"/>

  <img src="https://github.com/user-attachments/assets/125f5155-cfad-4597-8e79-3fd8e778043b" alt="qty_vs_price_scatter_시금치(일반)" width="250"/>

  <img src="https://github.com/user-attachments/assets/d8ff34a5-904e-49bd-b9e4-28e19d91ba20" alt="qty_vs_price_scatter_애호박" width="250"/>

</div>


* **주간 총액/반입량 추이**(품목별 시간경과에 따른 추세·계절성 시각화)

<div style="display: flex; justify-content: center; gap: 10px;">

  <img src="https://github.com/user-attachments/assets/afa38bee-6aad-4cbb-849f-992d02162044" alt="trends_amount_week_item_시금치(일반)" width="300"/>

  <img src="https://github.com/user-attachments/assets/5f11ea92-03ae-448c-9df1-992b0b8a4492" alt="trends_price_week_item_시금치(일반)" width="300"/>

  <img src="https://github.com/user-attachments/assets/6a7d4ca5-1510-4b92-8e7a-ba9c5a2e38b0" alt="trends_qty_week_item_시금치(일반)" width="300"/>

</div>


---

## 2. 피처 엔지니어링 (`src/features/make_weekly_features.py`)

**목적**
주간 집계(`weekly_agg.parquet`)에 **기상·캘린더·랙/롤링·변동성** 피처를 체계적으로 추가하고, **다음 주 단가**를 타깃으로 하는 학습 입력 `data/features/weekly_features.parquet`를 생성합니다.

**기상 데이터 주간화**

* 인코딩 자동 시도: `utf-8-sig → utf-8 → cp949`
* 날짜 컬럼 유연 처리(예: `"조회일자"(YYYYMMDD) → date` 변환)
* 표준 변수: `tavg, tmin, tmax, precip, humidity, sunshine, radiation, wind`
* **W-MON 리샘플**: 강수·일조·일사=주합(sum), 나머지=주평균(mean)

**시계열 피처 설계(누설 방지)**

* 로그 변환: `log_price = log1p(unit_price_week)`, `log_qty = log1p(qty_in_week)`
* **랙**: L ∈ {1,2,4,8,12,52} → `log_price_lag{L}`, `log_qty_lag{L}`
* **롤링**: **`shift(1)` 적용 후** W ∈ {4,8,12,26,52} → `*_rollmeanW`, `*_rollstdW`
* **연간 차이**: `log_price_yoy_gap = log_price - log_price_lag52`
* **변동성**: `price_vol_W = std(pct_change(unit_price_week))` with `shift(1).rolling(W)`, W ∈ {4,8,12}
* **캘린더**: `year`, `month`, `weekofyear(ISO)`
* **타깃**: `next_week_unit_price`, `next_week_log_price` = `shift(-1)`

**데이터 품질 & 안전장치**

* 모든 롤링은 **반드시 `shift(1)`** 후 계산(누설 방지)
* 랙/롤링 초기결측 증가에 대한 드롭/대치 정책 문서화
* 그룹 내 `week` 기준 정렬 보장, 기상 결합 누락 시 보수적 처리

**보고서용 추천 시각화**

* 피처 중요도(SHAP/Permutation), 상관 히트맵(품목별), 변동성 타임라인(`price_vol_12` 상위 구간 표시), **기상–단가 시차 관계**(예: `tmax_lag{1..8}` vs `log_price`)

---

## 3. 베이스라인: SNaive(52주 계절)

**정의**
다음 주(t+1) 예측값 = **작년 같은 주((t+1)-52)의 실측 단가**.
검증 기간 고정(2023-01-01 ~ 2023-12-31)으로 전 모델 공정 비교.

**산출물**

* 예측(검증): `predictions/val/snaive_val.csv`
* 전체 지표: `reports/metrics/baseline_val_metrics.csv`
* 품목별 지표: `reports/metrics/baseline_val_by_item.csv`

**지표**(핵심: WAPE, sMAPE)

* WAPE, sMAPE, MAE, n

---

## 4. 모델 개발 · 최적화

### 4-1) 글로벌 GBM 회귀(LightGBM, 로그 타깃)

**설정 요약**

* **타깃**: `next_week_log_price` (log1p 변환)
* **피처**: 랙/롤링(가격·수량), 변동성(`price_vol_*`), 캘린더, 주간화 기상, `item`(카테고리)
* **학습/검증 분리**: Train ≤ 2022-12-31, Valid = 2023-01-01~2023-12-31
* **파라미터(예)**: `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=30`, `feature_fraction=0.85`, `subsample=0.9`, `seed=42`
* **산출물**

  * 모델: `models/artifacts/global_lgbm.pkl`
  * 검증 예측: `predictions/val/global_lgbm_val.csv`
  * 지표: `reports/metrics/global_lgbm_val_metrics.csv`, `..._val_by_item.csv`
  * 중요도: `reports/metrics/global_lgbm_feature_importance.csv`

**홀드아웃(2023) 성능 요약**

* **WAPE = 0.181**, **sMAPE = 0.170**, **MAE = 726.4**, **RMSE_log = 0.239**
* 베이스라인 대비 **개선율**: WAPE **43.4%**, sMAPE **47.8%**, MAE **43.4%**

  * (Baseline: WAPE 0.320, sMAPE 0.326, MAE 1,283.7)

> 비교표(요약)

| 모델             |      WAPE |     sMAPE |       MAE |
| -------------- | --------: | --------: | --------: |
| SNaive(52)     |     0.320 |     0.326 |   1,283.7 |
| **Global GBM** | **0.181** | **0.170** | **726.4** |

**피처 중요도(상위, gain 기준)**
(파일: `reports/metrics/global_lgbm_feature_importance.csv`)

* 상위 예: `log_price`, `log_price_rollmean4`, `item`, `log_price_lag1`, `weekofyear`, `log_price_lag52`, `log_price_lag2`, `log_price_yoy_gap`, `log_price_rollmean52`, `qty_in_week`, `log_qty_yoy_gap`, `price_vol_4`

![feature\_importance\_gain](reports/figures/interpret/feature_importance_gain.png)

---

### 4-2) 연도 단위 Rolling/Blocked 교차검증

* 폴드: 2019→2020, …, 2023→2024 (총 5폴드), 각 검증 12개월
* 결과(요약): 폴드별 **WAPE ~ 0.184 → 0.174**, 평균 **WAPE ≈ 0.181**, **sMAPE ≈ 0.168**, **RMSE_log ≈ 0.235**
  → **홀드아웃과 유사**, 연도 변동성에도 **일반화 안정성 확보**

---

### 4-3) 품목별 성능 진단 (`..._val_by_item.csv`)

* **상위(낮은 WAPE) 예시**:
  `마늘 (0.055); 깐양파 (0.058); 표고버섯(일반) (0.061); 새송이(일반) (0.077); 수미 (0.085)`
* **하위(높은 WAPE) 예시**:
  `치콘 (0.259); 파프리카 (0.274); 영양부추 (0.275); 치커리(일반) (0.290); 쪽파(일반) (0.354)`

> 하위 품목군은 **공급 충격/기상 민감/데이터 희소** 가능성이 커, **대체재 추천 강화 및 룰 최적화** 우선 적용을 권장합니다.

---

## 5. 대체재 추천 알고리즘 (`src/recommenders/suggest.py`)

**목적**
가격 급등 리스크 발생 시 **동일 품목군 내**에서 **저가·공급 안정·낮은 변동성** 후보를 자동 추천합니다.

**입력**

* 예측: `predictions/next_week/global_lgbm_next_week.csv`
* 최근 실측: `data/processed/weekly_agg.parquet`
* 규칙: `suggestions/substitution_rules.yaml`

  * `rules`: `price_discount_threshold`(기본 0.15), `top_k`(기본 5)
  * `aliases`: 표기 표준화 사전
  * `groups`: 품목군 정의(예: 엽경채류, 버섯류 등)

**로직**

1. 기준 주 데이터 결합(예측가/이번주가/공급량)
2. 안정성 지표: `qty_rollmean4`(최근 4주 평균 공급), `price_vol_8`(최근 8주 변동성)
3. 군별 기준치: 공급 **중앙값**, 변동성 **상위 25% 분위**
4. 필터링: **절감율 ≥ τ**, **공급량 ≥ 중앙값**, **변동성 ≤ Q3**
5. 정렬: `saving_rate ↓`, `price_vol_8 ↑`(안정성 고려)
6. 출력: `suggestions/suggestions_next_week.csv` (품목별 Top-K)

**예시(시금치 → 대체재)**
절감율 ≥ 15%, 공급/변동성 기준 충족 시
→ `근대`, `청경채` 등 **동일 군** 후보를 우선 추천

**운영 KPI**

* 추천 커버리지(%) · 평균 절감율(%) · 안정성 비율(%) · 수용율(%)

---

## 6. 배포/운영 파이프라인

**목표**
최신 주까지 갱신된 피처를 입력으로 **다음 주 단가 예측 CSV**를 산출하고, 대시보드에 반영합니다.

**주요 스크립트**

* 다음 주 예측: `src/models/predict_next_week.py`
  → `predictions/next_week/global_lgbm_next_week.csv`
* 대체재 추천: `src/recommenders/suggest.py`
  → `suggestions/suggestions_next_week.csv`

**스케줄**
매주 **월요일 00:30** 배치 실행(갱신 → 예측 → 추천 → 대시보드)

**운영상 유의사항**

* 예측 대상 선정 로직: `next_week_unit_price` 결측 행(일부 품목 누락 가능)
  → **“품목별 최신 주 1행 강제 선택”** 보완 권장
* 파일 잠금·경로 이슈 대응: 안전 저장 유틸 적용 권장

---

## 7. 재현(리프로듀서블) 커맨드

```bash
# 1) 원시 통합 → 주간 집계 → 피처 생성
python -m src.data.build_master_raw
python -m src.data.build_weekly
python -m src.features.make_weekly_features

# 2) 베이스라인
python -m src.models.train_baselines

# 3) Global GBM (수정본)
python -m src.models.train_global_gbm

# 4) (선택) 튜닝 / 퀀타일 / 잔차 리포트
python -m src.models.tune_global_gbm_optuna --trials 60 --cv True
python -m src.models.train_global_gbm_quantile
python -m src.reports.plot_residuals_by_item

# 5) (운영) 다음 주 예측 & 대체재 추천
python -m src.models.predict_next_week
python -m src.recommenders.suggest
```

---

## 8. 프로젝트 결과 요약

1. **데이터 인사이트**

* 주간 단가 시계열에서 **계절성·수급효과·기상 영향(특히 tmax 3–4주 랙)** 확인

2. **예측 모델**

* 베이스라인: SNaive(52주)
* **Global GBM**: **WAPE 0.181 / sMAPE 0.170** (2023 홀드아웃)
  → 베이스라인 대비 **WAPE 43.4%**, **sMAPE 47.8%** 개선
  → 연도 단위 CV에서도 유사 성능(일반화 안정성 확보)

3. **대체재 추천**

* 동일 군 내 **저가·공급 안정·저변동성** 품목 Top-K 추천
* YAML 규칙 기반으로 **비개발자도 운영 파라미터 조정 가능**

4. **운영/확장성**

* 주간 배치 → 예측·추천 자동 산출
* 향후: Optuna 최적화 반영, **퀀타일 회귀(리스크 밴드)** 배포, 로컬 보정, **영양/조리 제약** 통합, 조달 API 연동

---

## 9. 저장소 산출물(발췌)

* 예측/지표

  * `predictions/val/snaive_val.csv`
  * `predictions/val/global_lgbm_val.csv`
  * `predictions/next_week/global_lgbm_next_week.csv`
  * `reports/metrics/baseline_val_metrics.csv` / `..._by_item.csv`
  * `reports/metrics/global_lgbm_val_metrics.csv` / `..._by_item.csv` / `global_lgbm_cv_metrics.csv` / `global_lgbm_feature_importance.csv`
* 시각화

  * `reports/figures/weekly_eda/coverage_heatmap_top40.png`
  * `reports/figures/weekly_eda/price_boxplot_top30.png`
  * `reports/figures/weekly_eda/qty_vs_price_scatter_all.png`
  * `reports/figures/interpret/feature_importance_gain.png`

---

## 10. 라이선스 / 문의

* (필요 시) 라이선스 고지
* 문의: 팀 이메일 또는 이슈 트래커

---

### 부록: 해석 가이드 예문

* “SNaive는 **계절성만** 활용하므로, 이 모델의 WAPE가 낮은 품목은 **규칙적 계절 패턴**이 강함을 의미합니다.”
* “Global GBM은 **자가 시계열 구조(가격 랙·롤링)**와 **수급(수량)**, **계절(weekofyear)**, **기상**을 함께 반영하여 베이스라인 대비 **유의미한 개선**을 달성했습니다.”
* “하위 품목은 **공급 충격·기상 민감·자료 희소** 영향이 크며, **대체재 추천 강화** 및 **규칙 파라미터 조정**이 효과적입니다.”

---

> **주**: 본 README의 수치·이미지 경로는 저장소 내 실제 산출물을 기반으로 확인했습니다. 저장소 구조가 변경되면 경로를 함께 갱신해 주세요.
