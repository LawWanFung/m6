"""
統計 / 序列趨勢模型
=================

不依赖 sklearn 分類框架，直接輸出「下一期每個號碼出現機率」的分佈，
再用 rolling-origin（滾動原點）的 log-loss 評估。

包含：
  * :class:`FrequencyModel`  — 頻率法 / 熱號策略（賭徒謬誤）
  * :class:`MarkovTrendModel` — 馬爾可夫鏈：每個號碼的 2 狀態轉移趨勢

⚠️ 六合彩係獨立隨機事件，這些模型都唔會真的贏過均勻隨機。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Predictor, parse_numbers, uniform_baseline


class FrequencyModel(Predictor):
    """
    頻率法（熱號策略 / 賭徒謬誤）。

    假設「長週期出現較多次的號碼，下一期也較可能出现」——這正係典型嘅
    賭徒謬誤。實際上下期各號碼機率恒為均勻，此模型會輸給均勻基線。

    用加一平滑（Laplace）避免零機率，prob = (count + 1) / (total + 49)。
    """

    name = "頻率法 熱號 Frequency"
    kind = "統計"
    desc = "長週期熱號策略（賭徒謬誤）：用歷史頻率當機率。"

    def _next_proba(self, counts: np.ndarray) -> np.ndarray:
        proba = (counts + 1.0) / (counts.sum() + 49)
        return proba

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        nums = parse_numbers(df)
        counts = np.zeros(49, dtype=float)
        losses = []
        last_proba = None
        for t in range(len(nums)):
            if t > 0:
                counts += np.bincount(nums[t - 1], minlength=50)[:49]
            proba = self._next_proba(counts)
            winning = nums[t]
            losses.append(-np.mean(np.log(proba[winning - 1] + 1e-12)))
            last_proba = proba
        mean_loss = float(np.mean(losses))
        top = np.argsort(last_proba)[::-1][:6].tolist() if last_proba is not None else None
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "predicted_numbers": [int(n) for n in sorted(top)] if top is not None else None,
        }


class MarkovTrendModel(Predictor):
    """
    馬爾可夫鏈 趨勢模型。

    為每個號碼建立一個 2 狀態（存在 / 不存在）的一階馬爾可夫鏈，
    估計「由上一期的狀態， transitions 到本期存在的機率」：
      * 若上期存在 → P(繼續存在)
      * 若上期不存在 → P(轉為存在)

    即捕捉「熱號持續、冷號翻生」的轉移趨勢，再正規化成機率分佈。
    """

    name = "馬爾可夫鏈 趨勢 Markov"
    kind = "統計"
    desc = "每個號碼的 2 狀態轉移：捕捉存在/不存在的持續與翻生趨勢。"

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        nums = parse_numbers(df)

        def proba_from_last(T):
            """依最後一期每個號碼的上一狀態，算出下期各號碼存在的機率分佈。"""
            cur_present = set(nums[-1].tolist())
            pp = np.zeros(49)
            for i in range(1, 50):
                a = 1 if i in cur_present else 0
                rr = T[a].sum()
                pp[i - 1] = T[a, 1] / rr
            total = pp.sum()
            return pp / total if total > 0 else np.full(49, 1.0 / 49)

        # --- 下一期預測（用全部歷史）：儀表板顯示用 ---
        T_full = np.full((2, 2), 1.0)  # 加一平滑
        for prev, cur in zip(nums, nums[1:]):
            for i in cur:
                T_full[int((i not in prev)), 1] += 1.0
            for i in set(range(1, 50)) - set(cur.tolist()):
                T_full[int((i not in prev)), 0] += 1.0
        _display_proba = proba_from_last(T_full)
        top = np.argsort(_display_proba)[::-1][:6].tolist()

        # --- rolling-origin 評估（增量更新轉移計數，避免 O(n²））---
        T = np.full((2, 2), 1.0)
        losses = []
        for t in range(1, len(nums)):
            # 用「截至 t-1 的轉移」預測第 t 期
            p = proba_from_last(T)
            losses.append(-np.mean(np.log(p[nums[t] - 1] + 1e-12)))
            # 增量加入 (t-1 -> t) 的轉移
            prev, cur = nums[t - 1], nums[t]
            for i in cur:
                T[int((i not in prev)), 1] += 1.0
            for i in set(range(1, 50)) - set(cur.tolist()):
                T[int((i not in prev)), 0] += 1.0
        mean_loss = float(np.mean(losses))
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "predicted_numbers": [int(n) for n in sorted(top)],
        }
