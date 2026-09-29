# -*- coding: utf-8 -*-
from pathlib import Path

# 프로젝트 루트 (네 환경에 맞게 바꾸고, 나머지는 자동)
PROJECT_ROOT = Path(r"D:\TMD_final_agri1")  # <- 네 PC 경로에 맞게 필요시 수정

# ── 입력 경로 ─────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PRICE_FINAL_CSV    = DATA_RAW_DIR / "price_final.csv"        # << 추가: 단일 원천 CSV
WEATHER_CSV        = DATA_RAW_DIR / "일별_기상데이터.csv"     # 기상 데이터
DATA_DROP_DIR      = DATA_RAW_DIR / "drop"                   # (구) 품목별 CSV 폴더(호환용)

# ── 중간/출력 경로 ────────────────────────────────────────────────────────────
DATA_INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FEATURE_DIR   = PROJECT_ROOT / "data" / "features"

MASTER_RAW_CSV     = DATA_PROCESSED_DIR / "master_raw.csv"
WEEKLY_AGG_PQ      = DATA_PROCESSED_DIR / "weekly_agg.parquet"
WEEKLY_FEAT_PQ     = DATA_FEATURE_DIR / "weekly_features.parquet"

MODELS_DIR         = PROJECT_ROOT / "models" / "artifacts"
GLOBAL_LGBM_PKL    = MODELS_DIR / "global_lgbm.pkl"

PRED_DIR           = PROJECT_ROOT / "predictions"
PRED_VAL_DIR       = PRED_DIR / "val"
PRED_CV_DIR        = PRED_DIR / "cv"
PRED_TEST_DIR      = PRED_DIR / "test"
PRED_NEXT_DIR      = PRED_DIR / "next_week"

REPORTS_FIG_DIR    = PROJECT_ROOT / "reports" / "figures"
REPORTS_MET_DIR    = PROJECT_ROOT / "reports" / "metrics"

SUGG_DIR           = PROJECT_ROOT / "suggestions"
SUGG_RULES_YAML    = SUGG_DIR / "substitution_rules.yaml"
SUGG_NEXT_CSV      = SUGG_DIR / "suggestions_next_week.csv"

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_dirs():
    for d in [
        DATA_INTERIM_DIR, DATA_PROCESSED_DIR, DATA_FEATURE_DIR,
        MODELS_DIR, PRED_VAL_DIR, PRED_CV_DIR, PRED_TEST_DIR, PRED_NEXT_DIR,
        REPORTS_FIG_DIR, REPORTS_MET_DIR, SUGG_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
