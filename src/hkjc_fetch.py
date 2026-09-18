"""
HKJC Mark Six 數據抓取模組
==========================

數據來源：香港賽馬會官方 JSON API
    GET http://bet.hkjc.com/marksix/getJSON.aspx?sd=YYYYMMDD&ed=YYYYMMDD&sb=0

特點：
  * 官方、權威、完整（可追溯到 1993 年）
  * 每次查詢最多約 3 個月，因此需要按 3 個月分段遍歷整段歷史
  * 返回純 JSON 陣列，欄位清晰（見下方 IRecord）

用法：
    python src/hkjc_fetch.py --from 1993-01-01 --to 2024-12-31 --out data/raw
    python src/hkjc_fetch.py --latest            # 只抓最近一期（增量更新）

注意：
  * 請遵守 HKJC 服務條款，控制請求頻率（預設每次請求暫停 0.8 秒）。
  * 若你的網絡環境封鎖 HKJC，需在能直連該網站的機器上執行。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

HOST = "http://bet.hkjc.com"
BASE_URL = f"{HOST}/marksix/getJSON.aspx"
# 官方限制每次查詢約 3 個月範圍
WINDOW_DAYS = 90
DEFAULT_DELAY = 0.8  # 秒，避免請求過頻

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def _iter_date_range(start: dt.date, end: dt.date):
    """將 [start, end] 切成多個不超過 WINDOW_DAYS 的區間。"""
    cur = start
    while cur <= end:
        nxt = cur + dt.timedelta(days=WINDOW_DAYS)
        if nxt > end:
            nxt = end
        yield cur, nxt
        cur = nxt + dt.timedelta(days=1)


def fetch_window(sd: dt.date, ed: dt.date, delay: float = DEFAULT_DELAY) -> list[dict]:
    """查詢單個區間，返回 JSON 陣列。"""
    params = {"sd": sd.strftime("%Y%m%d"), "ed": ed.strftime("%Y%m%d"), "sb": 0}
    try:
        resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"  [!] 請求失敗 {sd}~{ed}: {e}", file=sys.stderr)
        return []

    if resp.status_code != 200:
        print(f"  [!] HTTP {resp.status_code} {sd}~{ed}", file=sys.stderr)
        return []

    try:
        data = resp.json()
    except ValueError:
        print(f"  [!] 非 JSON 回應 ({len(resp.content)} bytes) {sd}~{ed}", file=sys.stderr)
        return []

    if delay:
        time.sleep(delay)
    return data or []


def normalize(rec: dict) -> dict:
    """將官方 JSON 正規化為統一欄位。"""
    numbers = [int(x) for x in rec.get("no", "").split("+") if x.strip()]
    return {
        "draw_id": rec.get("id"),
        "date": rec.get("date"),          # DD/MM/YYYY
        "numbers": numbers,               # 6 個開獎號碼
        "extra_ball": int(rec["sno"]) if rec.get("sno") else None,  # 特別號
        "snowball_code": rec.get("sbcode"),
        "snowball_name_en": rec.get("sbnameE"),
        "snowball_name_ch": rec.get("sbnameC"),
        "total_turnover": _comma_to_float(rec.get("inv")),
        "p1": _comma_to_float(rec.get("p1")),
        "p1_units": _comma_to_float(rec.get("p1u")),
        "p2": _comma_to_float(rec.get("p2")),
        "p2_units": _comma_to_float(rec.get("p2u")),
        "p3": _comma_to_float(rec.get("p3")),
        "p4": _comma_to_float(rec.get("p4")),
        "p5": _comma_to_float(rec.get("p5")),
        "p6": _comma_to_float(rec.get("p6")),
        "p7": _comma_to_float(rec.get("p7")),
    }


def _comma_to_float(v):
    if v in (None, "", "-"):
        return None
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


def fetch_history(start: dt.date, end: dt.date, out_dir: Path, delay: float = DEFAULT_DELAY) -> int:
    """遍歷整段歷史，將結果逐區間寫入 data/raw，並返回總筆數。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    windows = list(_iter_date_range(start, end))
    print(f"共 {len(windows)} 個區間，開始抓取 {start} ~ {end} ...")
    for i, (sd, ed) in enumerate(windows, 1):
        recs = fetch_window(sd, ed, delay)
        # 每區間存一個 json 檔，方便斷點續傳
        seg = out_dir / f"seg_{sd.strftime('%Y%m%d')}_to_{ed.strftime('%Y%m%d')}.json"
        seg.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        total += len(recs)
        print(f"  [{i}/{len(windows)}] {sd} ~ {ed}: {len(recs)} 筆 -> {seg.name}")
    print(f"完成，共 {total} 筆。")
    return total


def fetch_latest(out_dir: Path, delay: float = DEFAULT_DELAY) -> int:
    """只抓取最近 30 天（增量更新）。"""
    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    return fetch_history(start, end, out_dir, delay=delay)


def parse_args():
    p = argparse.ArgumentParser(description="HKJC Mark Six 數據抓取")
    p.add_argument("--from", dest="start", help="起始日期 YYYY-MM-DD（預設 1993-01-01）")
    p.add_argument("--to", dest="end", help="截止日期 YYYY-MM-DD（預設今日）")
    p.add_argument("--out", default="data/raw", help="原始 JSON 輸出目錄")
    p.add_argument("--latest", action="store_true", help="只抓最近增量（30 天）")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="每次請求延遲秒數")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = Path(a.out)
    if a.latest:
        fetch_latest(out_dir, a.delay)
        return
    start = dt.date.fromisoformat(a.start) if a.start else dt.date(1993, 1, 1)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    fetch_history(start, end, out_dir, a.delay)


if __name__ == "__main__":
    main()
