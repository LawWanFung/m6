"""
建模入口（single-model demo：MLP）
==================================

⚠️ 六合彩係獨立隨機事件，任何基於歷史號碼的模型都無法真正預測下一期。
本模塊的目標係「示範流程」並「證實無法預測」：

我們訓練一個模型，用過去 N 期的號碼特徵去預測**下期**各號碼出現的機率，
然後對比「模型預測」與「均勻隨機」的表現。結果你會見到模型表現
並不比隨機好 —— 這正係六合彩不可預測的統計證據。

包含：
  * 特徵工程：滯後頻率（過去 lookback 期每個號碼出現次數）
  * 使用 scikit-learn 訓練分類器（預測 1-49 每個號碼的機率）
  * 與 baseline（均勻分佈）比較 log-loss

⚠️ 關鍵點（舊版本嘅三個缺陷，已修正）：
  1. 舊版本用 `TimeSeriesSplit(n_splits=3).split(X)` 按**樣本索引**切 → 同一
     攪珠期嘅 6 條樣本共用同一特徵向量，切點處同一向量會同時出現在訓練同評估
     集（泄漏）。現在按**攪珠期**切（`draw_folds`）。
  2. 舊版本喺 49 個類別循環裡 `clf.predict_proba(X[te])` 重算 49 次 → 極慢。
     現在每 fold 只算一次。
  3. 訓練集未覆蓋的號碼列是 0 → log_loss 把它 clip 成 EPS，得到無意義的大值；
     現在填 EPS 並歸一化，同時記錄 `class_coverage`。

用 API 時傳 `write_result=False`，避免每個 HTTP 請求都重寫 model/results 文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - 有些 stdout 不可 reconfigure
    pass
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analyze.frequency as freqmod
from model.base import (
    parse_numbers,
    multiclass_samples,
    draw_folds,
    pick_draws,
    uniform_baseline,
    window_features,
)
from model.ml_models import _proba_matrix

RESULTS_DIR = BASE / "model" / "results"


def make_features(df: pd.DataFrame, lookback: int = 10):
    """為每一期建構特徵，目標 = 當期 6 個主號（多分類，49 類）。

    返回 `(X, y, draw_idx)`；無樣本時 `(None, None, None)`。
    """
    return multiclass_samples(df, lookback)


def run(df: pd.DataFrame, lookback: int = 10, write_result: bool = True) -> dict:
    nums = parse_numbers(df)
    X, y, draw_idx = make_features(df, lookback)

    if X is None or len(X) == 0:
        msg = f"數據不足：需要多於 lookback={lookback} 期（現得 {len(nums)} 期）"
        print(f"[!] {msg}")
        return {
            "lookback": lookback,
            "n_samples": 0,
            "n_draws": int(len(df)),
            "data_source": df.attrs.get("data_source", "unknown"),
            "mean_log_loss": None,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": False,
            "error": msg,
        }

    print(f"訓練樣本: {len(X)} 個, 類別數: {len(np.unique(y))}")

    # 以「期」為單位限制規模：唔泄漏，亦令 /api/model 唔會每請求幾十秒
    folds = draw_folds(draw_idx, n_folds=3, train_frac=0.7,
                       max_train_draws=600, max_test_draws=200)
    baseline = uniform_baseline()
    losses = []
    for fold, (tr, te) in enumerate(folds):
        # 每折新造一個 pipeline（StandardScaler + MLP），唔共享狀態；
        # MLP 已標準化 + early_stopping，避免「未收斂」嘅 artifacts。
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=400, n_iter_no_change=10,
                early_stopping=True, random_state=0,
            )),
        ])
        clf.fit(X[tr], y[tr])
        proba, n_missing = _proba_matrix(clf, X[te])
        loss = float(log_loss(y[te], proba, labels=range(1, 50)))
        losses.append(loss)
        print(f"  fold {fold + 1}: log-loss = {loss:.4f}"
              + (f"（未覆蓋類別 {n_missing} 個）" if n_missing else ""))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    print(f"模型平均 log-loss: {mean_loss:.4f}  vs  均勻 baseline: {baseline:.4f}")

    # ── 下一期預測：用「最近 <=600 期」再訓練一個模型（控制耗時）。
    #    ⚠️ 特徵窗口必須用 t = len(nums)（預測尚未發生的那一期）；用
    #    len(nums) - 1 會重算最後一期已知號碼（「用答案算答案」），預測毫無意義。
    nums = parse_numbers(df)
    sample_draws = sorted({int(d) for d in draw_idx})
    by_draw: dict[int, list[int]] = {}
    for i, d in enumerate(draw_idx):
        by_draw.setdefault(int(d), []).append(i)
    tr = [i for d in pick_draws(sample_draws, 600) for i in by_draw[d]]
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=400, n_iter_no_change=10,
            early_stopping=True, random_state=0,
        )),
    ])
    clf.fit(X[tr], y[tr])

    predicted = None
    if len(nums) > lookback:
        x_next = window_features(nums, len(nums), lookback).reshape(1, -1)
        proba, _ = _proba_matrix(clf, x_next)
        top = np.argsort(proba[0])[::-1][:6]  # 保持概率順序，唔用 sorted()
        predicted = [int(n) + 1 for n in top]

    result = {
        "lookback": lookback,
        "n_samples": int(len(X)),
        "n_draws": int(len(df)),
        "data_source": df.attrs.get("data_source", "unknown"),
        "mean_log_loss": mean_loss,
        "uniform_baseline_log_loss": baseline,
        "can_predict": bool(mean_loss < baseline),
        "predicted_numbers": predicted,
        "n_folds": len(losses),
    }
    if write_result:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with (RESULTS_DIR / "model_result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"結果已存至 {RESULTS_DIR / 'model_result.json'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--csv", help="指定 CSV 路徑")
    ap.add_argument("--allow-sample", action="store_true",
                    help="容許在無真實數據時退回合成樣本（只供離線測試）")
    a = ap.parse_args()
    try:
        df = freqmod.load(Path(a.csv) if a.csv else None, allow_sample=a.allow_sample)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[!] 數據不可用: {exc}", file=sys.stderr)
        return 1
    run(df, a.lookback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
