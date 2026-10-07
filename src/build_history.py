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

# `hkjc_fetch` 係同目錄模塊。先把 src/ 加入 sys.path，令呢度喹任何 cwd /
# 任何導入方式（`python src/build_history.py`、`python -m src.build_history`、
# 被其他模塊 import）都能找到它。
BASE = Path(__file__).resolve().parent.parent
_SRC = str(BASE / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hkjc_fetch import (
    DEFAULT_DELAY,
    DEFAULT_LAST_N_DRAW,
    fetch_draws,
    is_result,
    normalize,
    _iter_date_range,  # noqa: F401  （保留給外部／舊介面使用）
)
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
    if not isinstance(data, list):
        print(f"  [!] 跳過 {path}：JSON 唔係陣列", file=sys.stderr)
        return []
    # ⚠️ 只有「已開獎」(status == "Result" 且有 drawnNo) 先可以入主表。
    # Pending／未開獎嘅記錄冇 numbers，入主表會令 analyze.load() 同 model
    # 的 parse_numbers 崩潰，而且會造成主表出現空號碼行。
    recs = []
    for r in data:
        if not isinstance(r, dict):
            continue
        # 已正規化過的記錄（有 draw_id、無 GraphQL 嘅 id/drawnNo 字段）直接收，
        # **唔可以**再 normalize：二次 normalize 會丟失 numbers → 空行。
        if "draw_id" in r and "id" not in r:
            if r.get("numbers"):
                recs.append(r)
            continue
        if r.get("id") and is_result(r):
            recs.append(normalize(r))
    return recs


def _rec_from_csv_row(row: dict) -> dict:
    """將主表 CSV 一行轉回記錄（numbers 字串 -> list[int]），供 merge 用。"""
    rec = dict(row)
    nums = row.get("numbers") or ""
    rec["numbers"] = [int(v) for v in str(nums).split(",") if str(v).strip().isdigit()]
    return rec


def _load_existing_csv() -> dict[str, dict]:
    """讀現有主表（已入 repo 嘅真實歷史）。"""
    out: dict[str, dict] = {}
    if not HISTORY_CSV.exists():
        return out
    try:
        with HISTORY_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                did = row.get("draw_id")
                if did:
                    out[did] = _rec_from_csv_row(row)
    except (OSError, ValueError) as e:
        print(f"  [!] 讀取現有主表失敗（忽略）：{e}", file=sys.stderr)
    return out


def build_history(merge_existing: bool = True) -> int:
    """合併「現有主表」+ 所有 raw JSON -> 主表 CSV（按 draw_id 去重）。

    ⚠️ 重要：必需以現有主表為底再疊上 raw 快取。
    因為 `data/raw/`（原始 JSON 快取）**唔入 repo**，如果單純用 raw 覆寫，
    在一個只有主表、冇 raw 快取嘅環境（新 clone／Docker）執行就會
    用少量甚至 1 筆資料**清空已累積嘅 4390 期歷史**。
    """
    seen: dict[str, dict] = _load_existing_csv() if merge_existing else {}
    base_n = len(seen)

    raw_n = 0
    if RAW_DIR.exists():
        for seg in sorted(RAW_DIR.glob("*.json")):
            for rec in _load_seg(seg):
                did = rec.get("draw_id")
                if not did:
                    continue
                # 唔盲目以 raw 為準：raw 有時只有號碼、冇 turnover／p* 獎金，
                # 直接覆寫會丟掉主表中較完整嘅記錄。逐字段合併，空值不覆蓋。
                seen[did] = _merge_rec(seen.get(did), rec)
                raw_n += 1
    else:
        print(f"  [i] 找不到 {RAW_DIR}（原始快取唔入 repo），只用現有主表。")

    if not seen:
        print("  [!] 冇任何資料可整合。請先執行 hkjc_fetch.py 抓取數據。", file=sys.stderr)
        return 0

    # 冇號碼（未開獎／損壞）的行會令下游解析崩潰，直接丟棄並提示
    bad = [r for r in seen.values() if not r.get("numbers")]
    if bad:
        print(f"  [!] 跳過 {len(bad)} 行冇開獎號碼（唔入主表）。", file=sys.stderr)
    rows = [r for r in seen.values() if r.get("date") and r.get("numbers")]
    rows.sort(key=_sort_key)  # 按真實日期升序，唔依賴 DD/MM/YYYY 字符串排序

    if not rows:
        print("[!] 沒有資料可整合（raw 目錄為空）。請先執行 hkjc_fetch.py 或 gen_sample.py。")
        return 0

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    # 寫臨時檔再取代：避免半個文件被其他讀取方（例如 /api/models）讀到
    tmp = HISTORY_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_row_for_csv(r))
    tmp.replace(HISTORY_CSV)

    print(
        f"已整合 {len(rows)} 筆 -> {HISTORY_CSV}"
        f"（現有主表 {base_n} 筆 + raw {raw_n} 筆，去重後 {len(rows)} 筆）"
    )
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


def _merge_rec(old: dict | None, new: dict) -> dict:
    """合併同一 draw_id 的記錄：新記錄為主，但空值唔會覆蓋舊記錄已有的值。

    主表可能已含有 total_turnover / p1..p7 等較完整嘅字段；raw JSON 有時冇這些，
    所以「以 raw 為準」嘅覆寫會造成數據損失。
    """
    if not old:
        return new
    merged = dict(old)
    for k, v in new.items():
        if v is None or v == "" or v == []:
            continue
        merged[k] = v
    return merged


def _sort_key(rec: dict):
    """把 DD/MM/YYYY 轉成可排序的 (年,月,日)。"""
    try:
        d, m, y = rec["date"].split("/")
        return (int(y), int(m), int(d))
    except (ValueError, KeyError):
        return (9999, 99, 99)


def update_latest(delay: float = DEFAULT_DELAY, last_n: int = DEFAULT_LAST_N_DRAW) -> int:
    """用 GraphQL 抓最近 last_n 期，合併進主表。

    合併方式 = 全量讀取 → 按 draw_id 去重（逐字段合併）→ 整檔按日期升序寫回
    （先寫臨時檔再 replace）。回傳新增筆數；`-1` = 抓取失敗或檔案損壞。

    ⚠️ 三件事必須做到，否則主表會被破壞：
      1. 抓取失敗 ≠ 該區間冇新攪珠；失敗時唔改主表，回傳 -1（上層可報錯）。
      2. 主表必須恆為日期升序：API 回傳「最新排最前」，而且補錄嘅攪珠
         （例如 Pending 遲遲先開獎）日期可能**舊於**主表現有尾段 —— 單純
         append 會把佢放尾、打亂時序；model/base.py 窗口特徵按行序取，
         倒序尾段 = 用「未來」做「過去」（泄漏／垃圾特徵）。整檔排序寫回先穩妥。
      3. 寫回前驗證 header 同 COLUMNS 一致（舊 schema 檔案會令字段漂移）。
    """
    fetched, status = _fetch_range(last_n, delay)
    if status == "error":
        print("[!] 增量抓取失敗（網絡／API 問題），主表未改動。", file=sys.stderr)
        return -1

    new_recs = [r for r in fetched if r.get("draw_id") and r.get("numbers")]
    if not new_recs:
        print("[i] 該區間冇新數據（或窗口過大被靜默截斷）；主表未改動。")
        return 0

    # 寫回前驗證 header（舊 schema 檔案若直接 read→merge，DictReader 會按舊
    # 欄名映射 → 字段漂移、資料靜默丟失）→ 直接中止，叫人手檢查。
    if HISTORY_CSV.exists():
        with HISTORY_CSV.open(encoding="utf-8") as f:
            first = f.readline().strip()
        if first != ",".join(COLUMNS):
            print(
                f"[!] 主表 header 同 COLUMNS 唔一致（舊 schema？）；"
                f"為避免字段漂移已中止增量合併。請檢查 {HISTORY_CSV}。",
                file=sys.stderr,
            )
            return -1

    existing = _load_existing_csv()
    before = len(existing)
    for rec in new_recs:
        existing[rec["draw_id"]] = _merge_rec(existing.get(rec["draw_id"]), rec)
    added = len(existing) - before

    # 冇號碼（未開獎／損壞）的行會令下游解析崩潰，直接丟棄並提示
    rows = [r for r in existing.values() if r.get("date") and r.get("numbers")]
    rows.sort(key=_sort_key)  # 整檔按真實日期升序（補錄嘅舊日期攪珠亦歸入正確位置）

    if not rows:
        print("[!] 合併後無有效行；主表未改動。", file=sys.stderr)
        return -1

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    # 先寫臨時檔再取代：避免半個文件被其他讀取方（例如 /api/models）讀到
    tmp = HISTORY_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_row_for_csv(r))
    tmp.replace(HISTORY_CSV)

    print(f"增量更新完成，新增 {added} 筆（整檔日期排序寫回，總計 {len(rows)} 筆）。")
    return added


def _fetch_range(last_n: int, delay: float) -> tuple[list[dict], str]:
    """經 GraphQL 抓最近 last_n 期並正規化（已開獎者）。回傳 (recs, status)。"""
    draws, status = fetch_draws(last_n, delay=delay)
    return [normalize(d) for d in draws if is_result(d) and d.get("id")], status


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "update"], default="build")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--last-n", type=int, default=DEFAULT_LAST_N_DRAW, help="增量抓取最近 N 期")
    a = p.parse_args()
    if a.mode == "build":
        return 0 if build_history() > 0 else 1
    return 0 if update_latest(a.delay, a.last_n) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
