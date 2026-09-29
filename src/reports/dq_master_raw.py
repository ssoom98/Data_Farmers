# -*- coding: utf-8 -*-
"""
Data Quality Report for master_raw (보고서 1-9 항목 자동화)
- 원천-표준 스키마 맵핑 표
- 인코딩 분포 요약(막대 그래프)
- 결측/형변환 영향 요약(막대)
- 품목 수/기간 커버리지 요약 + 커버리지 히트맵
- 행 수 추이(원천 총행 → 변환 후 총행) 비교 바차트

산출:
 - reports/tables/dq/column_mapping.csv
 - reports/tables/dq/encoding_summary.csv
 - reports/tables/dq/missing_and_casting_impact.csv
 - reports/tables/dq/coverage_summary.csv
 - reports/figures/dq/encoding_bar.png
 - reports/figures/dq/missing_casting_bar.png
 - reports/figures/dq/rowcount_bar.png
 - reports/figures/dq/coverage_heatmap_top40.png
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트(선택): 환경에 설치되어 있으면 자동 적용
try:
    import koreanize_matplotlib  # noqa: F401
except Exception:
    pass

from src.config import (
    PRICE_FINAL_CSV, DATA_DROP_DIR, MASTER_RAW_CSV, ensure_dirs
)
# 기존 빌드 로직 재사용(컬럼 후보 목록 포함)
from src.data.build_master_raw import REQ_COLS

FIG_DIR = Path("reports/figures/dq")
TAB_DIR = Path("reports/tables/dq")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = ["utf-8-sig", "utf-8", "cp949"]

def try_read(path: Path):
    """인코딩 자동 탐지 + DataFrame 반환, 성공 인코딩명 포함"""
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV 읽기 실패: {path} (마지막 에러: {last_err})")

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ensure_dirs()

    # 0) 입력 후보 수집
    inputs = []
    if Path(PRICE_FINAL_CSV).exists():
        inputs = [Path(PRICE_FINAL_CSV)]
    else:
        inputs = sorted([p for p in Path(DATA_DROP_DIR).glob("*.csv")])

    if not inputs:
        raise FileNotFoundError(
            f"입력 없음: price_final.csv도 없고, drop 폴더에도 CSV가 없습니다.\n"
            f"- 기대 경로 1) {PRICE_FINAL_CSV}\n- 기대 경로 2) {Path(DATA_DROP_DIR) / '*.csv'}"
        )

    # 1) 파일별 인코딩/맵핑/결측-형변환 영향 로깅
    map_rows = []
    enc_rows = []
    miss_rows = []

    total_rows_before = 0
    total_rows_after = 0

    parts_after = []  # 변환 후(part-level) 누적 for coverage

    for p in inputs:
        df_raw, enc = try_read(p)
        enc_rows.append({"file": str(p), "encoding": enc, "n_rows": len(df_raw)})

        total_rows_before += len(df_raw)

        # 컬럼 매핑
        c_date   = pick_col(df_raw, REQ_COLS["date"])
        c_item   = pick_col(df_raw, REQ_COLS["item"])
        c_qty    = pick_col(df_raw, REQ_COLS["qty_in"])
        c_amount = pick_col(df_raw, REQ_COLS["amount"])

        map_rows.append({
            "file": str(p),
            "date_src": c_date,
            "item_src": c_item,
            "qty_in_src": c_qty,
            "amount_src": c_amount,
            "date_std": "date" if c_date else None,
            "item_std": "item" if c_item else None,
            "qty_in_std": "qty_in" if c_qty else None,
            "amount_std": "amount" if c_amount else None,
        })

        # 형변환 + 결측 영향(파일 단위)
        tmp = pd.DataFrame({
            "date":   pd.to_datetime(df_raw[c_date], errors="coerce") if c_date else pd.NaT,
            "item":   df_raw[c_item].astype(str).str.strip() if c_item else np.nan,
            "qty_in": pd.to_numeric(df_raw[c_qty], errors="coerce") if c_qty else np.nan,
            "amount": pd.to_numeric(df_raw[c_amount], errors="coerce") if c_amount else np.nan,
        })
        # 드롭 전 결측 집계
        n_date_na  = int(tmp["date"].isna().sum())
        n_item_na  = int(tmp["item"].isna().sum())
        n_qty_nan  = int(pd.isna(tmp["qty_in"]).sum())
        n_amt_nan  = int(pd.isna(tmp["amount"]).sum())

        # 파일별 변환 후 유효 행 (date/item not null)
        tmp2 = tmp.dropna(subset=["date", "item"]).sort_values(["item","date"]).reset_index(drop=True)
        total_rows_after += len(tmp2)
        parts_after.append(tmp2)

        miss_rows.append({
            "file": str(p),
            "n_rows_before": len(df_raw),
            "drop_date_null": n_date_na,
            "drop_item_null": n_item_na,
            "qty_in_nan_after_cast": n_qty_nan,
            "amount_nan_after_cast": n_amt_nan,
            "n_rows_after_date_item_drop": len(tmp2),
        })

    # 2) 표 저장: 맵핑/인코딩/결측영향
    map_df = pd.DataFrame(map_rows)
    enc_df = pd.DataFrame(enc_rows)
    miss_df = pd.DataFrame(miss_rows)

    map_df.to_csv(TAB_DIR / "column_mapping.csv", index=False, encoding="utf-8-sig")
    enc_df.to_csv(TAB_DIR / "encoding_summary.csv", index=False, encoding="utf-8-sig")
    miss_df.to_csv(TAB_DIR / "missing_and_casting_impact.csv", index=False, encoding="utf-8-sig")

    # 3) 품목 수/기간 커버리지 (변환 후 기준)
    if parts_after:
        merged = pd.concat(parts_after, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["date","item","qty_in","amount"])
    # 커버리지 요약 표
    if not merged.empty:
        item_nunique = merged["item"].nunique()
        date_min = merged["date"].min()
        date_max = merged["date"].max()

        cover_rows = [{
            "n_items": int(item_nunique),
            "date_min": str(pd.to_datetime(date_min).date()) if pd.notna(date_min) else None,
            "date_max": str(pd.to_datetime(date_max).date()) if pd.notna(date_max) else None,
            "n_days_total": int((date_max - date_min).days + 1) if pd.notna(date_min) and pd.notna(date_max) else None,
            "n_rows_after": int(len(merged)),
        }]
        cov_df = pd.DataFrame(cover_rows)
    else:
        cov_df = pd.DataFrame([{"n_items":0,"date_min":None,"date_max":None,"n_days_total":0,"n_rows_after":0}])

    cov_df.to_csv(TAB_DIR / "coverage_summary.csv", index=False, encoding="utf-8-sig")

    # ========== Figures ==========
    # F1. 인코딩 분포 막대
    plt.figure(figsize=(8, 4))
    enc_counts = enc_df["encoding"].value_counts().sort_index()
    enc_counts.plot(kind="bar")
    plt.title("인코딩 분포(파일 수)")
    plt.xlabel("encoding"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "encoding_bar.png", dpi=160)
    plt.close()

    # F2. 결측/형변환 영향(파일별)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    idx = np.arange(len(miss_df))
    width = 0.23
    ax.bar(idx - width, miss_df["drop_date_null"], width, label="date 결측 드롭")
    ax.bar(idx,          miss_df["drop_item_null"], width, label="item 결측 드롭")
    ax.bar(idx + width,  miss_df["qty_in_nan_after_cast"], width, label="qty_in NaN(형변환)")
    ax.bar(idx + 2*width,miss_df["amount_nan_after_cast"], width, label="amount NaN(형변환)")
    ax.set_title("결측/형변환 영향(파일별)")
    ax.set_xlabel("file index"); ax.set_ylabel("rows")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_casting_bar.png", dpi=160)
    plt.close()

    # F3. 행 수 추이(원천 합 vs 변환 후 합)
    plt.figure(figsize=(6, 4))
    values = [total_rows_before, total_rows_after]
    labels = ["원천 총행", "변환 후 총행(date/item 유효)"]
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=10)
    plt.title("행 수 추이(원천 → 변환 후)")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rowcount_bar.png", dpi=160)
    plt.close()

    # F4. 일자 커버리지 히트맵 (상위 40개 품목)
    #  - 일자별 거래 유무(0/1)를 피벗. 품목수가 많으면 상위 40개(행 많은 순)만 그림
    if not merged.empty:
        merged["day"] = merged["date"].dt.date
        # 품목별 등장 일수(행수)
        top_items = (merged.groupby("item")["day"].count()
                     .sort_values(ascending=False).head(40).index.tolist())
        df_top = merged[merged["item"].isin(top_items)].copy()
        # presence 0/1
        df_top["presence"] = 1
        mat = (df_top.pivot_table(index="item", columns="day", values="presence",
                                  aggfunc="max", fill_value=0)
                      .sort_index())
        # 그리기 (이미지 크기 자동 조정)
        h = max(6, 0.18 * len(mat))
        w = max(10, 0.04 * len(mat.columns))
        plt.figure(figsize=(w, h))
        plt.imshow(mat.values, aspect="auto", interpolation="nearest")
        plt.colorbar(label="coverage (0/1)")
        plt.yticks(range(len(mat.index)), mat.index)
        # x축 눈금이 너무 많으면 생략 간격
        step = max(1, len(mat.columns)//30)
        xticks = np.arange(0, len(mat.columns), step)
        xticklabels = [str(pd.to_datetime(c).date()) for i, c in enumerate(mat.columns) if i % step == 0]
        plt.xticks(xticks, xticklabels, rotation=90)
        plt.title("일자 커버리지 히트맵 (상위 40 품목)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "coverage_heatmap_top40.png", dpi=180)
        plt.close()

    print("[OK] DQ 리포트 생성 완료")
    print(" - Tables:", TAB_DIR)
    print(" - Figures:", FIG_DIR)

if __name__ == "__main__":
    main()
