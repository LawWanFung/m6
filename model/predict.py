"""
建模基底（含重要提醒）
=====================

⚠️ 六合彩係獨立隨機事件，任何基於歷史號碼的模型都無法真正預測下一期。
本模組的目標係「示範流程」並「證實無法預測」：

我們訓練一個模型，用過去 N 期的號碼特徵去预测下期各號碼出現的機率，
然後對比「模型預測」與「均勻隨機」的表現。結果你會見到模型表現
並不比隨機好 —— 這正係六合彩不可預測的統計證據。

包含：
  * 特徵工程：滯後頻率（過去 lookback 期每個號碼出現次數）
  * 使用 scikit-learn 訓練分類器（預測 1-49 每個號碼的機率）
  * 與 baseline（均勻分佈）比較 log-loss
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

import analyze.frequency as freqmod
from model.base import parse_numbers

RESULTS_DIR = BASE / "model" / "results"


def make_features(df: pd.DataFrame, lookback: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    為每一期建構特徵，目標 = 當期 6 個主號（多分類，49 類）。
    特徵：過去 lookback 期每個號碼的出現次數 (lookback x 49)。
    """
    nums = parse_numbers(df)
    X, y = [], []
    for t in range(lookback, len(nums)):
        window = np.concatenate([
            np.bincount(nums[t - k], minlength=50)[1:50]
            for k in range(1, lookback + 1)
        ])  # (lookback * 49,)
        # 每個「(期, 號碼)」為一個樣本，各獲一份特徵
        for n in nums[t]:
            X.append(window.copy())
            y.append(n)
    return np.vstack(X), np.array(y)


def run(df: pd.DataFrame, lookback: int = 10) -> dict:
    X, y = make_features(df, lookback)
    print(f"訓練樣本: {len(X)} 個, 類別數: {len(np.unique(y))}")

    tscv = TimeSeriesSplit(n_splits=3)
    losses = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0)
        clf.fit(X[tr], y[tr])
        classes = clf.classes_  # 訓練集的類別（可能少於 49 個）
        proba = np.zeros((len(te), 49))
        for j, c in enumerate(classes):
            proba[:, c - 1] = clf.predict_proba(X[te])[:, j]
        loss = log_loss(y[te], proba, labels=range(1, 50))
        losses.append(loss)
        print(f"  fold {fold + 1}: log-loss = {loss:.4f}")

    mean_loss = float(np.mean(losses))
    baseline = float(np.log(49))  # 均勻分佈的 log-loss
    print(f"模型平均 log-loss: {mean_loss:.4f}  vs  均勻 baseline: {baseline:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "lookback": lookback,
        "n_samples": int(len(X)),
        "n_draws": int(len(df)),
        "data_source": df.attrs.get("data_source", "unknown"),
        "mean_log_loss": mean_loss,
        "uniform_baseline_log_loss": baseline,
        "can_predict": mean_loss < baseline,  # 通常為 False
    }
    with (RESULTS_DIR / "model_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"結果已存至 {RESULTS_DIR / 'model_result.json'}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--csv", help="指定 CSV 路徑")
    ap.add_argument("--allow-sample", action="store_true",
                    help="容許在無真實數據時退回合成樣本（只供離線測試）")
    a = ap.parse_args()
    df = freqmod.load(Path(a.csv) if a.csv else None, allow_sample=a.allow_sample)
    run(df, a.lookback)


if __name__ == "__main__":
    main()
