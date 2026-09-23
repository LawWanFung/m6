"""
建模基底（共用特徵工程與評估框架）
=================================

⚠️ 六合彩係**獨立隨機事件**，任何基於歷史號碼的模型都**無法真正預測**下一期。
本模組的目標係「示范流程」並「證實無法預測」：每個模型輸出一個 1–49 嘅機率分佈，
再用同一個 metric（log-loss）對比「均勻基線（ln 49 ≈ 3.89）」，
結果模型通常輸給隨機 —— 呢個正正係「六合彩不可預測」嘅統計證據。

設計成可插拔的 registry：每個模型係一個 :class:`Predictor` 子類，
實作 :meth:`Predictor.evaluate` 即可以喺儀表板統一列出所有模型結果。

包含：
  * :func:`parse_numbers`        — 解析 CSV 嘅號碼欄
  * :func:`window_features`      — 滯後頻率特徵（lookback × 49）
  * :func:`multiclass_samples`   — 多分類樣本（每個 (期, 號碼) 一個樣本）
  * :func:`uniform_baseline`     — 均勻基線 log-loss = ln(49)
  * :class:`Predictor`           — 模型基底介面
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

#: 均勻分佈的 log-loss 理論值 = ln(49)，所有模型嘅對比基線。
UNIFORM_BASELINE = float(np.log(49))
#: log 計算時嘅小保護值，避免 log(0)。
EPS = 1e-12


def parse_numbers(df: pd.DataFrame) -> List[np.ndarray]:
    """將 `numbers` 欄解析成每期一個的 int numpy 陣列。

    相容兩種格式：
      * "1,8,13,24,35,43"   （sample CSV / 原始 CSV）
      * "[1, 8, 13, 24, 35, 43]"  （主表 CSV，含方括號）
    """
    out = []
    for v in df["numbers"]:
        s = str(v).strip().strip("[]")
        nums = sorted(int(x) for x in s.split(","))
        out.append(np.array(nums, dtype=int))
    return out


def window_features(nums: List[np.ndarray], t: int, lookback: int) -> np.ndarray:
    """
    建構第 `t` 期嘅特徵 = 對上 `lookback` 期、每個號碼（1–49）出現嘅次數。

    返回 shape `(lookback * 49,)`。
    """
    feats = [
        np.bincount(nums[t - k], minlength=50)[1:50]
        for k in range(1, lookback + 1)
    ]
    return np.concatenate(feats)


def multiclass_samples(df: pd.DataFrame, lookback: int):
    """
    將「預測下期各號碼機率」轉成多分類問題：

    對每個 `t >= lookback`，以第 `t` 期嘅滯後頻率特徵為輸入，
    第 `t` 期嘅每個主號為一個正樣本（label = 該號碼）。
    即係每個 ``(期, 號碼)`` 都是一條樣本，訓練集好大。

    返回 ``(X, y)``，X shape ``(n_samples, lookback*49)``。
    """
    nums = parse_numbers(df)
    X, y = [], []
    for t in range(lookback, len(nums)):
        x = window_features(nums, t, lookback)
        for n in nums[t]:
            X.append(x)
            y.append(n)
    return np.vstack(X), np.array(y)


def uniform_baseline() -> float:
    """均勻分佈的 log-loss 理論值 = ln(49)。"""
    return UNIFORM_BASELINE


class Predictor:
    """
    所有預測模型的統一介面。

    子類需要設定：
      * `name`   — 顯示名稱
      * `desc`   — 一句說明
      * `kind`   — 分類標籤（如 sklearn / 統計 / 深度學習）

    並實作 :meth:`evaluate`，返回儀表板要用的 dict。
    """

    name: str = "base"
    desc: str = ""
    kind: str = "misc"

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        raise NotImplementedError

    @staticmethod
    def verdict(mean_log_loss: float) -> bool:
        """模型「能贏隨機」的判定：log-loss 低於均勻基線才算。"""
        return bool(mean_log_loss < UNIFORM_BASELINE)
