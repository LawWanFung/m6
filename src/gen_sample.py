"""
生成樣本數據（合成、可重現）
============================

因為本環境封鎖 HKJC 官方網站，本檔案生成一份「合成」樣本，方便本地測試
分析 / 模型流程。實際落地時，请用 hkjc_fetch.py 抓真實數據後覆寫此檔。

特性：
  * 500 期，每期 6 個不重複號碼 (1-49) + 1 個特別號 (1-49, 可與主號重疊)
  * 使用固定 random seed，確保每次生成結果一致（可重現）
  * 日期由 2020-01-01 起，逢 二/四/六 開獎
"""

from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

sys = __import__("sys")
sys.stdout.reconfigure(encoding="utf-8")

SEED = 20240520
N = 500
BASE = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE / "data" / "mark6_sample.csv"

DRAW_DAYS = [1, 3, 5]  # 二(1)/四(3)/六(5) — Python weekday(): Mon=0
DAY_CYCLE = [1, 3, 5]


def generate() -> list[dict]:
    rng = random.Random(SEED)
    start = dt.date(2020, 1, 7)  # 2020-01-07 係星期一後 first Tuesday
    rows = []
    d = start
    di = 0
    while len(rows) < N:
        # 跳到下一個開獎日
        while d.weekday() != DAY_CYCLE[di % len(DAY_CYCLE)]:
            d += dt.timedelta(days=1)
        di += 1
        main = rng.sample(range(1, 50), 6)
        extra = rng.randint(1, 49)
        rows.append({
            "draw_id": f"SYN{len(rows)+1:05d}",
            "date": d.strftime("%d/%m/%Y"),
            "numbers": main,
            "extra_ball": extra,
            "snowball_code": "",
            "snowball_name_ch": "",
            "total_turnover": round(rng.uniform(40e6, 70e6), 2),
            "p1": round(rng.uniform(0.5e6, 5e6), 2),
            "p1_units": round(rng.uniform(20, 300), 2),
            "p2": round(rng.uniform(30e3, 300e3), 2),
            "p2_units": round(rng.uniform(200, 2000), 2),
            "p3": round(rng.uniform(10e3, 100e3), 2),
            "p4": round(rng.uniform(3e3, 30e3), 2),
            "p5": round(rng.uniform(1e3, 10e3), 2),
            "p6": round(rng.uniform(500, 5e3), 2),
            "p7": round(rng.uniform(200, 2e3), 2),
        })
        d += dt.timedelta(days=1)
    return rows


def main():
    rows = generate()
    COLUMNS = ["draw_id", "date", "numbers", "extra_ball", "snowball_code",
               "snowball_name_ch", "total_turnover", "p1", "p1_units", "p2",
               "p2_units", "p3", "p4", "p5", "p6", "p7"]
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (",".join(map(str, r[k])) if k == "numbers" else r[k]) for k in COLUMNS})
    print(f"已生成 {len(rows)} 筆合成樣本 -> {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
