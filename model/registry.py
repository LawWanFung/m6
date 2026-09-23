"""
預測模型 Registry（統一入口）
==========================

把所有模型註冊成一個有序列表，供 app 的 `/api/models` 端點與儀表板統一列出。

⚠️ 六合彩係獨立隨機事件，所有模型都唔會真的贏過均勻隨機基線（ln 49 ≈ 3.89）。

使用：
    from model.registry import run_all, MODELS
    results = run_all(df, lookback=10)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Predictor, uniform_baseline
from .ml_models import (
    LogisticRegressionModel,
    MLPModel,
    GaussianNBModel,
    RandomForestModel,
)
from .stat_models import FrequencyModel, MarkovTrendModel


class UniformBaseline(Predictor):
    """均勻基線：不訓練、不評估，只作對比參考（log-loss = ln(49) ≈ 3.89）。"""

    name = "均勻基線 Uniform"
    kind = "baseline"
    desc = "理論參考線：各號碼機率恒等，log-loss = ln(49) ≈ 3.89。"

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        b = uniform_baseline()
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": b,
            "uniform_baseline_log_loss": b,
            "can_predict": False,
            "predicted_numbers": None,
        }


#: 所有模型（順序即儀表板顯示順序）。基線永遠排第一。
MODELS: List[Predictor] = [
    UniformBaseline(),
    LogisticRegressionModel(),
    MLPModel(),
    RandomForestModel(),
    GaussianNBModel(),
    FrequencyModel(),
    MarkovTrendModel(),
]


def run_all(df: pd.DataFrame, lookback: int = 10) -> list:
    """對全部模型執行 evaluate，返回有序結果列表（含均勻基線）。"""
    results = []
    for m in MODELS:
        try:
            results.append(m.evaluate(df, lookback))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "name": m.name,
                "kind": m.kind,
                "description": m.desc,
                "mean_log_loss": None,
                "uniform_baseline_log_loss": uniform_baseline(),
                "can_predict": False,
                "error": str(exc),
            })
    return results


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    import analyze.frequency as freqmod

    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--csv", help="指定 CSV 路徑")
    ap.add_argument("--allow-sample", action="store_true")
    a = ap.parse_args()
    df = freqmod.load(Path(a.csv) if a.csv else None, allow_sample=a.allow_sample)
    res = run_all(df, a.lookback)
    for r in res:
        print(f"{r['name']:<28} log-loss={r['mean_log_loss']}")
