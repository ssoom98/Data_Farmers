# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from pathlib import Path
from src.config import (
    PRICE_FINAL_CSV, DATA_DROP_DIR, MASTER_RAW_CSV, ensure_dirs
)

REQ_COLS = {
    "date": ["거래일자", "date"],
    "item": ["품목명", "품목", "item"],
    "qty_in": ["반입량", "수량", "qty_in"],
    "amount": ["금액", "amount"],
}

def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 읽기 실패: {path}")

def _pick_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_from_price_final(p: Path) -> pd.DataFrame:
    raw = _read_csv_any(p)

    c_date   = _pick_col(raw, REQ_COLS["date"])
    c_item   = _pick_col(raw, REQ_COLS["item"])
    c_qty    = _pick_col(raw, REQ_COLS["qty_in"])
    c_amount = _pick_col(raw, REQ_COLS["amount"])

    missing = [k for k,(c) in {
        "거래일자": c_date, "품목명": c_item, "반입량": c_qty, "금액": c_amount
    }.items() if c is None]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (입력: {p})")

    out = pd.DataFrame({
        "date":   pd.to_datetime(raw[c_date], errors="coerce"),
        "item":   raw[c_item].astype(str).str.strip(),
        "qty_in": pd.to_numeric(raw[c_qty], errors="coerce"),
        "amount": pd.to_numeric(raw[c_amount], errors="coerce"),
    })
    out = out.dropna(subset=["date", "item"]).sort_values(["item","date"]).reset_index(drop=True)
    return out

def main():
    ensure_dirs()

    if Path(PRICE_FINAL_CSV).exists():
        df = build_from_price_final(Path(PRICE_FINAL_CSV))
    else:
        # (호환 모드) 구 방식: drop 폴더의 품목별 CSV들을 모아서 생성
        files = sorted([p for p in Path(DATA_DROP_DIR).glob("*.csv")])
        if not files:
            raise FileNotFoundError(
                f"입력 없음: price_final.csv도 없고, drop 폴더에도 CSV가 없습니다.\n"
                f"- 기대 경로 1) {PRICE_FINAL_CSV}\n- 기대 경로 2) {DATA_DROP_DIR}\\*.csv"
            )
        parts = []
        for p in files:
            raw = _read_csv_any(p)
            c_date   = _pick_col(raw, REQ_COLS["date"])
            c_qty    = _pick_col(raw, REQ_COLS["qty_in"])
            c_amount = _pick_col(raw, REQ_COLS["amount"])
            missing = [k for k,(c) in {
                "거래일자": c_date, "반입량": c_qty, "금액": c_amount
            }.items() if c is None]
            if missing:
                raise ValueError(f"필수 컬럼 누락: {missing} (입력: {p})")

            part = pd.DataFrame({
                "date":   pd.to_datetime(raw[c_date], errors="coerce"),
                "item":   p.stem,  # 파일명=품목명 가정 (구 방식)
                "qty_in": pd.to_numeric(raw[c_qty], errors="coerce"),
                "amount": pd.to_numeric(raw[c_amount], errors="coerce"),
            })
            parts.append(part)

        df = pd.concat(parts, ignore_index=True).dropna(subset=["date"]).sort_values(["item","date"])

    df.to_csv(MASTER_RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] master_raw.csv 저장 → {MASTER_RAW_CSV} (rows={len(df)})")

if __name__ == "__main__":
    main()
