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


# ---------------------------------------------------------------------------
# 號碼歷史上場窗口（已實證）
#   1–45  由首期（1993-01-05）起
#   46–47 由 1996-06-11 起  （1996-06-11 增至 47 個號碼）
#   48–49 由 2002-07-04 起  （2002-07-04 增至 49 個號碼，48/49 正式加入）
#
# 後加嘅號碼（46-49）若用全歷史 raw count 排冷熱，會因為「出現得少」
# 而被誤判為冷號。因此冷熱改用『率』（每期出現機率）排名，
# 並按資格窗口正規化。
_ELIGIBILITY_START = {n: pd.Timestamp("1993-01-05") for n in range(1, 46)}
_ELIGIBILITY_START[46] = pd.Timestamp("1996-06-11")
_ELIGIBILITY_START[47] = pd.Timestamp("1996-06-11")
_ELIGIBILITY_START[48] = pd.Timestamp("2002-07-04")
_ELIGIBILITY_START[49] = pd.Timestamp("2002-07-04")

# 時代分界（對應上面的資格窗口，供卡方分時代檢驗）
ERAS = [
    ("1993-1996", pd.Timestamp("1993-01-05"), pd.Timestamp("1996-06-10"), 45),
    ("1996-2002", pd.Timestamp("1996-06-11"), pd.Timestamp("2002-07-03"), 47),
    ("2002-至今", pd.Timestamp("2002-07-04"), None, 49),
]


def eligibility_windows() -> dict:
    """回傳 {號碼: 資格起始日(ISO)}，方便 UI 顯示。"""
    return {int(k): v.date().isoformat() for k, v in sorted(_ELIGIBILITY_START.items())}


def eligibility_stats(df: pd.DataFrame) -> dict:
    """回傳 {num: {count, eligible_draws, rate}}。

    每個號碼只計入其有資格（date >= 資格起始日）嘅期數，
    rate = 出現次數 / 符合資格期數（正規化後嘅『率』）。
    """
    dates = df["date"]
    stats_ = {}
    for n in range(1, 50):
        start = _ELIGIBILITY_START[n]
        mask = dates >= start
        eligible = int(mask.sum())
        if eligible == 0:
            stats_[n] = {"count": 0, "eligible_draws": 0, "rate": 0.0}
            continue
        counts = df.loc[mask, "numbers"]
        count = int(sum(sum(1 for x in nums if x == n) for nums in counts))
        stats_[n] = {"count": count, "eligible_draws": eligible, "rate": count / eligible}
    return stats_


def frequency_analysis(df: pd.DataFrame) -> dict:
    """主號 1-49 與特別號頻率。

    除咗 raw count（main_freq / extra_freq），额外提供：
      - main_rate：按資格窗口正規化之後嘅『率』（每期出現機率）
      - eligibility：每個號碼嘅 count / eligible_draws / rate
    """
    all_main = [n for nums in df["numbers"] for n in nums]
    main_series = pd.Series(all_main)
    main_freq = main_series.value_counts().reindex(range(1, 50)).fillna(0)

    extra = df["extra_ball"].dropna()
    extra_freq = pd.Series(extra).value_counts().reindex(range(1, 50)).fillna(0)

    elig = eligibility_stats(df)
    main_rate = pd.Series({n: elig[n]["rate"] for n in range(1, 50)})

    return {
        "n_numbers": int(main_series.count()),
        "main_count": int(main_series.sum()),
        "main_expected": int(main_series.sum() / 49),
        "main_freq": main_freq,
        "extra_freq": extra_freq,
        "main_rate": main_rate,
        "eligibility": elig,
    }


def chi_square_uniform(freq: pd.Series, label: str, total: int):
    """檢驗 freq 是否服從均勻分佈（pool 內等機率）。

    pool 大小由 freq 的長度決定（== reindex 的 range 長度），
    因此分時代檢驗時傳入該時代嘅 pool 即可。
    """
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


def _era_main_chi(df_era: pd.DataFrame, pool: int, label: str) -> dict:
    """某個時代主號的卡方检验（pool 固定，才有意義）。"""
    main_counts = pd.Series(
        [n for nums in df_era["numbers"] for n in nums]
    ).value_counts().reindex(range(1, pool + 1)).fillna(0)
    total = int(main_counts.sum())
    return chi_square_uniform(main_counts, label, total)


def _era_extra_chi(df_era: pd.DataFrame, pool: int, label: str) -> dict:
    """某個時代特別號的卡方检验。"""
    extra_counts = pd.Series(df_era["extra_ball"].dropna()).value_counts().reindex(
        range(1, pool + 1)
    ).fillna(0)
    total = int(extra_counts.sum())
    return chi_square_uniform(extra_counts, label, total)


def chi_square_era(df: pd.DataFrame) -> list:
    """分時代检验均匀性。

    全歷史把所有 49 個號碼放一齊測，會因「後加號碼」出现较少而产生
    假性「不均勻」。拆成三代各測各代嘅 pool（45 / 47 / 49），先有意义。
    """
    results = []
    for era_name, start, end, pool in ERAS:
        if end is None:
            era_df = df[df["date"] >= start]
        else:
            era_df = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(era_df) == 0:
            continue
        results.append({
            "era": era_name,
            "pool": pool,
            "n_draws": int(len(era_df)),
            "main_chi2": _era_main_chi(era_df, pool, era_name),
            "extra_chi2": _era_extra_chi(era_df, pool, era_name),
        })
    return results


def _empty_chi() -> dict:
    return {"label": "n/a", "chi2": None, "p_value": None, "dof": None,
            "max_dev": None, "significant_05": False}


def cold_hot(rate: pd.Series, top: int = 5):
    """按『率』排名（已按資格窗口正規化），唔係 raw count。

    冷熱用 rate（每期出現機率），所以遲出嘅號碼（46-49）唔會被誤判為冷。
    """
    ordered = rate.sort_values(ascending=False)
    return {
        "hot": {int(k): round(float(v), 4) for k, v in ordered.head(top).items()},
        "cold": {int(k): round(float(v), 4) for k, v in ordered.tail(top).iloc[::-1].items()},
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
    era_results = chi_square_era(df)
    # UI 嘅 main_chi2 / extra_chi2 用「最新一代」（當前 49 個號碼條件下）
    main_chi = era_results[-1]["main_chi2"] if era_results else _empty_chi()
    extra_chi = era_results[-1]["extra_chi2"] if era_results else _empty_chi()
    ch = cold_hot(freq["main_rate"])
    desc = descriptive_stats(df)

    out = {
        "n_draws": int(len(df)),
        "data_source": source,          # "real" = 真實 HKJC 數據；"sample" = 合成樣本
        "data_path": path,
        "date_range": {
            "first": str(df["date"].min().date()) if df["date"].notna().any() else None,
            "last": str(df["date"].max().date()) if df["date"].notna().any() else None,
        },
        "eligibility_windows": eligibility_windows(),
        "main_chi2": main_chi,
        "extra_chi2": extra_chi,
        "era_chi2": era_results,
        "cold_hot": {k: {int(k2): v for k2, v in val.items()} for k, val in ch.items()},
        "descriptive": desc,
        "main_freq": {int(k): int(v) for k, v in freq["main_freq"].items()},
        "main_rate": {int(k): round(float(v), 4) for k, v in freq["main_rate"].items()},
        "extra_freq": {int(k): int(v) for k, v in freq["extra_freq"].items()},
    }
    with (RESULTS_DIR / "analysis.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    # 列印摘要
    print(f"數據期數: {out['n_draws']}（來源: {source} · {path}）")
    if out["date_range"]["first"]:
        print(f"日期範圍: {out['date_range']['first']} ~ {out['date_range']['last']}")
    if era_results:
        for e in era_results:
            mc = e["main_chi2"]
            print(f"  {e['era']}（pool {e['pool']}）主號 χ² p={mc['p_value']:.4f} "
                  f"(显著0.05:{mc['significant_05']}) · 特別號 p={e['extra_chi2']['p_value']:.4f}")
    else:
        print("主號 卡方: 無數據")
    print(f"熱號(率): { {str(k): v for k, v in ch['hot'].items()} }")
    print(f"冷號(率): { {str(k): v for k, v in ch['cold'].items()} }")
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
