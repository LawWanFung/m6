"""
數據整合 / 增量更新
===================

1. build_history()   : 將 data/raw 下所有 JSON 合併成一份主表
                       data/mark6_history.csv（按 date 排序，draw_id 去重）。
2. update_latest()   : 經 GraphQL 抓最近增量，只追加主表中尚無的記錄（按 draw_id 去重）。

注意：GraphQL API 一次最多約 58 期（見 docs/hkjc_graphql_schema.md），
所以「完整歷史」係靠本主表長期累積，唔可能一次過由 1993 年撈到最新。

主表欄位：
    draw_id, date, numbers(6), extra_ball, snowball_code, snowball_name_ch,
    total_turnover, p1..p7 (+ _units)
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

from hkjc_fetch import (
    DEFAULT_DELAY,
    DEFAULT_LAST_N_DRAW,
    fetch_draws,
    is_result,
    normalize,
    _iter_date_range,  # noqa: F401  （保留給外部／舊介面使用）
)

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
HISTORY_CSV = BASE / "data" / "mark6_history.csv"

COLUMNS = [
    "draw_id", "date", "numbers", "extra_ball", "snowball_code",
    "snowball_name_ch", "total_turnover", "p1", "p1_units", "p2", "p2_units",
    "p3", "p4", "p5", "p6", "p7",
]


def _load_seg(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        print(f"  [!] 跳過損壞檔案 {path}", file=sys.stderr)
        return []
    return [normalize(r) for r in data if r.get("id")]


def build_history() -> int:
    """合併所有區間 JSON -> 主表 CSV（去重）。"""
    if not RAW_DIR.exists():
        print(f"[!] 找不到 {RAW_DIR}", file=sys.stderr)
        return 0

    seen: dict[str, dict] = {}
    for seg in sorted(RAW_DIR.glob("*.json")):
        for rec in _load_seg(seg):
            did = rec["draw_id"]
            if did:
                seen[did] = rec  # 相同 draw_id 以後來為準

    rows = sorted(
        (r for r in seen.values() if r.get("date")),
        key=lambda r: r["date"],  # DD/MM/YYYY，排序僅供參考，見 _sort_key
    )
    rows.sort(key=_sort_key)

    if not rows:
        print("[!] 沒有資料可整合（raw 目錄為空）。請先執行 hkjc_fetch.py 或 gen_sample.py。")
        return 0

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_row_for_csv(r))

    print(f"已整合 {len(rows)} 筆 -> {HISTORY_CSV}")
    return len(rows)


def _row_for_csv(rec: dict) -> dict:
    """轉成 CSV 可寫格式（numbers 必須以逗號字串輸出，否則 analyze.load() 解析唔到）。"""
    out = {k: rec.get(k, "") for k in COLUMNS}
    nums = rec.get("numbers")
    if isinstance(nums, (list, tuple)):
        out["numbers"] = ",".join(str(n) for n in nums)
    for k, v in out.items():
        if v is None:
            out[k] = ""
    return out


def _sort_key(rec: dict):
    """把 DD/MM/YYYY 轉成可排序的 (年,月,日)。"""
    try:
        d, m, y = rec["date"].split("/")
        return (int(y), int(m), int(d))
    except (ValueError, KeyError):
        return (9999, 99, 99)


def _existing_draw_ids() -> set[str]:
    """讀取現有主表的 draw_id 集合。"""
    ids = set()
    if not HISTORY_CSV.exists():
        return ids
    with HISTORY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(row["draw_id"])
    return ids


def update_latest(delay: float = DEFAULT_DELAY, last_n: int = DEFAULT_LAST_N_DRAW) -> int:
    """用 GraphQL 抓最近 last_n 期，只追加新記錄到主表。"""
    existing = _existing_draw_ids()
    added = 0
    new_rows: list[dict] = []

    for rec in _fetch_range(last_n, delay):
        did = rec["draw_id"]
        if did and did not in existing:
            new_rows.append(rec)
            existing.add(did)
            added += 1

    if new_rows:
        HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if HISTORY_CSV.exists() else "w"
        with HISTORY_CSV.open(mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if mode == "w":
                w.writeheader()
            for r in new_rows:
                w.writerow(_row_for_csv(r))
    print(f"增量更新完成，新增 {added} 筆。")
    return added


def _fetch_range(last_n: int, delay: float) -> list[dict]:
    """經 GraphQL 抓最近 last_n 期並正規化（已開獎者）。"""
    draws = fetch_draws(last_n, delay=delay)
    return [normalize(d) for d in draws if is_result(d) and d.get("id")]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "update"], default="build")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--last-n", type=int, default=DEFAULT_LAST_N_DRAW, help="增量抓取最近 N 期")
    a = p.parse_args()
    if a.mode == "build":
        build_history()
    else:
        update_latest(a.delay, a.last_n)


if __name__ == "__main__":
    main()
