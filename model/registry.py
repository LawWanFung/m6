"""
預測模型 Registry（統一入口）
==========================

把所有模型註冊成一個有序列表，供 app 的 `/api/models` 端點與儀表板統一列出。

⚠️ 六合彩係獨立隨機事件，所有模型都不會真的贏過均勻隨機基線（ln 49 ≈ 3.89182）。
比較時要用 **實際值** `uniform_baseline_log_loss`，唔可以 hardcode 3.8918。

使用：
    from model.registry import run_all, make_models
    results = run_all(df, lookback=10)

也可以直接執行：
    python -m model.registry --lookback 10
    python model/registry.py --lookback 10        # 已支援（把 repo root 加入 sys.path）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

import pandas as pd  # noqa: F401 (型別註解 + 與 evaluate 簽名一致)

#: 把 repo root 加入 sys.path，令 `python model/registry.py` 同 `import model.registry` 都 works。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - 有些 stdout 不可 reconfigure
    pass

from model.base import Predictor, uniform_baseline
from model.ml_models import (
    LogisticRegressionModel,
    MLPModel,
    GaussianNBModel,
    RandomForestModel,
)
from model.stat_models import FrequencyModel, MarkovTrendModel


class UniformBaseline(Predictor):
    """均勻基線：不訓練、不評估，只作對比參考（log-loss = ln(49)）。"""

    name = "均勻基線 Uniform"
    kind = "baseline"
    desc = "理論參考線：各號碼機率恆等，log-loss = ln(49) ≈ 3.89182。"

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


#: 模型工廠（順序即儀表板顯示順序）。基線永遠排第一。
MODEL_FACTORIES: List[Callable[[], Predictor]] = [
    UniformBaseline,
    LogisticRegressionModel,
    MLPModel,
    RandomForestModel,
    GaussianNBModel,
    FrequencyModel,
    MarkovTrendModel,
]


def make_models() -> List[Predictor]:
    """每次調用都新造實例。

    ⚠️ 模型對象持有 `_last_clf` 等可變狀態。若用模塊級 singleton，併發 HTTP
    請求會共用同一個 estimator → 結果互相污染（同一 URL 兩次返回不同數字）。
    """
    return [factory() for factory in MODEL_FACTORIES]


#: 向後兼容的模塊級列表（運行時用 make_models()，唔依賴呢個共享列表）
MODELS: List[Predictor] = make_models()


def run_all(df: pd.DataFrame, lookback: int = 10) -> list:
    """對全部模型執行 evaluate，返回有序結果列表（含均勻基線）。"""
    results = []
    for m in make_models():
        try:
            results.append(m.evaluate(df, lookback))
        except Exception as exc:  # noqa: BLE001
            # 不靜默吞掉：寫進結果 + 打到 stderr，方便診斷（亦令 /api/models 唔會
            # 假裝「模型冇信號」而掩蓋真正的 bug）。
            print(f"[!] {m.name} 評估失敗: {exc}", file=sys.stderr)
            results.append({
                "name": m.name,
                "kind": m.kind,
                "description": m.desc,
                "mean_log_loss": None,
                "uniform_baseline_log_loss": uniform_baseline(),
                "can_predict": False,
                "predicted_numbers": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return results


def main() -> None:
    import argparse

    import analyze.frequency as freqmod

    ap = argparse.ArgumentParser(description="Run all prediction models on Mark Six data.")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--csv", help="指定 CSV 路徑")
    ap.add_argument("--allow-sample", action="store_true",
                    help="容許在無真實數據時退回合成樣本（只供離線測試）")
    a = ap.parse_args()

    try:
        df = freqmod.load(Path(a.csv) if a.csv else None, allow_sample=a.allow_sample)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[!] 數據不可用: {exc}", file=sys.stderr)
        raise SystemExit(1)

    res = run_all(df, a.lookback)
    for r in res:
        print(f"{r['name']:<32} log-loss={r['mean_log_loss']} "
              f"(baseline={r['uniform_baseline_log_loss']:.6f})")


if __name__ == "__main__":
    main()
