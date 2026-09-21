"""
六合彩統計分析
==============

載入歷史 CSV（若無則用合成樣本），進行：
  1. 號碼頻率分析（主號 1-49、特別號）
  2. 卡方檢驗：號碼分佈是否均勻（即是否「公平」/ 無偏）
  3. 冷熱號、連號、奇偶/大小比等描述性統計
  4. 結果持久化到 analyze/results/

所有「預測」僅為描述性展示，不構成任何預測意涵。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "mark6_history.csv"
SAMPLE_CSV = BASE / "data" / "mark6_sample.csv"
RESULTS_DIR = BASE / "analyze" / "results"


def load(path: Path | None = None, allow_sample: bool = False) -> pd.DataFrame:
    """載入主表 CSV。

    **預設只接受真實數據**（data/mark6_history.csv）：如果檔案不存在或為空，
    直接拋 `FileNotFoundError`，**絕不會靜默改用合成樣本**。

    只有明確傳 `allow_sample=True`（離線測試／CI）才會退回合成樣本，
    此時 `df.attrs["data_source"] == "sample"` 可以被呼叫方檢查。
    """
    target = Path(path) if path else CSV_PATH
    source = "real"

    def _empty() -> bool:
        try:
            return not target.exists() or target.stat().st_size == 0
        except OSError:
            return True

    if _empty():
        if allow_sample and SAMPLE_CSV.exists():
            print(f"[!] 找不到真實數據 {target}，已退回合成樣本 {SAMPLE_CSV.name}（僅供離線測試）")
            target = SAMPLE_CSV
            source = "sample"
        else:
            raise FileNotFoundError(
                f"找不到真實歷史數據：{target}\n"
                f"請先執行：\n"
                f"  python src/hkjc_fetch.py --from 1993-01-01   # 抓完整歷史\n"
                f"  python src/build_history.py --mode build     # 整合成主表"
            )

    df = pd.read_csv(target, encoding="utf-8")
    if len(df) == 0:
        if allow_sample and SAMPLE_CSV.exists() and target != SAMPLE_CSV:
            print(f"[!] 主表為空，已退回合成樣本 {SAMPLE_CSV.name}（僅供離線測試）")
            target = SAMPLE_CSV
            source = "sample"
            df = pd.read_csv(target, encoding="utf-8")
        else:
            raise ValueError(f"數據檔為空：{target}")

    df["numbers"] = df["numbers"].apply(lambda x: [int(v) for v in str(x).split(",") if v.strip()])
    df["extra_ball"] = pd.to_numeric(df["extra_ball"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    # 記下來源，方便 API / UI 顯示（並確保 demo 數據唔會被當成真實數據）
    df.attrs["data_source"] = source
    df.attrs["data_path"] = str(target)
    return df


def frequency_analysis(df: pd.DataFrame) -> dict:
    """主號 1-49 與特別號頻率。"""
    all_main = [n for nums in df["numbers"] for n in nums]
    main_series = pd.Series(all_main)
    main_freq = main_series.value_counts().reindex(range(1, 50)).fillna(0)

    extra = df["extra_ball"].dropna()
    extra_freq = pd.Series(extra).value_counts().reindex(range(1, 50)).fillna(0)

    return {
        "n_numbers": int(main_series.count()),
        "main_count": int(main_series.sum()),
        "main_expected": int(main_series.sum() / 49),
        "main_freq": main_freq,
        "extra_freq": extra_freq,
    }


def chi_square_uniform(freq: pd.Series, label: str, total: int):
    """檢驗 freq 是否服從均勻分佈（49 個號碼等機率）。"""
    observed = freq.values.astype(float)
    n = len(observed)
    expected = np.full(n, total / n)
    # 去除 expected 為 0 的項以避免警告
    mask = expected > 0
    chi2, p = stats.chisquare(f_obs=observed[mask], f_exp=expected[mask])
    dof = n - 1
    res = {
        "label": label,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "max_dev": float(np.max(np.abs(observed - expected))),
    }
    res["significant_05"] = bool(p < 0.05)
    return res


def cold_hot(freq: pd.Series, top: int = 5):
    ordered = freq.sort_values(ascending=False)
    return {
        "hot": ordered.head(top).to_dict(),
        "cold": ordered.tail(top).iloc[::-1].to_dict(),
    }


def descriptive_stats(df: pd.DataFrame) -> dict:
    """奇偶比、大小比(>=25為大)、連號數量等。"""
    stats_ = {}
    def _ratio(nums):
        even = sum(1 for n in nums if n % 2 == 0)
        big = sum(1 for n in nums if n >= 25)
        return even, big

    even_counts, big_counts, consecutive = [], [], []
    for nums in df["numbers"]:
        e, b = _ratio(nums)
        even_counts.append(e)
        big_counts.append(b)
        s = sorted(set(nums))
        cons = sum(1 for i in range(1, len(s)) if s[i] == s[i-1] + 1)
        consecutive.append(cons)

    return {
        "odd_even_mean": float(np.mean(even_counts)),  # 平均每個開獎日的大號(偶)個數
        "big_small_mean": float(np.mean(big_counts)),
        "consecutive_mean": float(np.mean(consecutive)),
        "sum_mean": float(np.mean([sum(n) for n in df["numbers"]])),
        "sum_std": float(np.std([sum(n) for n in df["numbers"]])),
    }


def run(df: pd.DataFrame) -> dict:
    """接收已載入的 DataFrame，執行分析並寫入 results/。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source = df.attrs.get("data_source", "unknown")
    path = df.attrs.get("data_path", "")
    if source == "sample":
        print("[!] 注意：今次分析用嘅係合成樣本，唔係真實 HKJC 數據！")

    freq = frequency_analysis(df)
    main_chi = chi_square_uniform(freq["main_freq"], "main_numbers", freq["n_numbers"])
    extra_chi = chi_square_uniform(freq["extra_freq"], "extra_ball", int(df["extra_ball"].dropna().count()))
    ch = cold_hot(freq["main_freq"])
    desc = descriptive_stats(df)

    out = {
        "n_draws": int(len(df)),
        "data_source": source,          # "real" = 真實 HKJC 數據；"sample" = 合成樣本
        "data_path": path,
        "date_range": {
            "first": str(df["date"].min().date()) if df["date"].notna().any() else None,
            "last": str(df["date"].max().date()) if df["date"].notna().any() else None,
        },
        "main_chi2": main_chi,
        "extra_chi2": extra_chi,
        "cold_hot": {k: {int(k2): int(v) for k2, v in val.items()} for k, val in ch.items()},
        "descriptive": desc,
        "main_freq": {int(k): int(v) for k, v in freq["main_freq"].items()},
        "extra_freq": {int(k): int(v) for k, v in freq["extra_freq"].items()},
    }
    with (RESULTS_DIR / "analysis.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    # 列印摘要
    print(f"數據期數: {out['n_draws']}（來源: {source} · {path}）")
    if out["date_range"]["first"]:
        print(f"日期範圍: {out['date_range']['first']} ~ {out['date_range']['last']}")
    print(f"主號 卡方 p={main_chi['p_value']:.4f} (显著性0.05: {main_chi['significant_05']}) "
          f"最大偏差 {main_chi['max_dev']:.1f}")
    print(f"特別號 卡方 p={extra_chi['p_value']:.4f}")
    print(f"熱號: { {str(k): v for k, v in ch['hot'].items()} }")
    print(f"冷號: { {str(k): v for k, v in ch['cold'].items()} }")
    print(f"結果已存至 {RESULTS_DIR / 'analysis.json'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="指定 CSV 路徑")
    ap.add_argument("--allow-sample", action="store_true",
                    help="容許在無真實數據時退回合成樣本（只供離線測試）")
    a = ap.parse_args()
    df = load(Path(a.csv) if a.csv else None, allow_sample=a.allow_sample)
    run(df)


if __name__ == "__main__":
    main()
