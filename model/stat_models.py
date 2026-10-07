"""
統計 / 序列趨勢模型
=================

不依賴 sklearn 分類框架，直接輸出「下一期每個號碼出現機率」的分佈，
再用 rolling-origin（滾動原點）的 log-loss 評估。

包含：
  * :class:`FrequencyModel`  — 頻率法 / 熱號策略（賭徒謬誤）
  * :class:`MarkovTrendModel` — 馬爾可夫鏈：每個號碼的 2 狀態轉移趨勢

⚠️ 六合彩係獨立隨機事件，這些模型都唔會真的贏過均勻隨機。

評估必須 causal：
  * 預測第 t 期時，只能用**第 t-1 期**（或更早）的信息。
  * `predicted_numbers` 係「下一期」嘅 top-6，按概率由高到低排列（唔用
    sorted() 打亂概率順序）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Predictor, parse_numbers, uniform_baseline


def _not_enough(name: str, kind: str, desc: str, need: int, have: int) -> dict:
    """數據不足時返回可渲染的「無結果」條目，唔會拋異常。"""
    return {
        "name": name,
        "kind": kind,
        "description": desc,
        "mean_log_loss": None,
        "uniform_baseline_log_loss": uniform_baseline(),
        "can_predict": False,
        "predicted_numbers": None,
        "error": f"數據不足：需要至少 {need} 期（現得 {have} 期）",
    }


class FrequencyModel(Predictor):
    """
    頻率法（熱號策略 / 賭徒謬誤）。

    假設「長週期出現較多次的號碼，下一期也較可能出現」——這正係典型嘅
    賭徒謬誤。實際上下期各號碼機率恆為均勻，此模型會輸給均勻基線。

    用加一平滑（Laplace）避免零機率：prob = (count + 1) / (total + 49)，
    分佈已歸一化（sum = 1），所以與 ln(49) 基線可比。
    """

    name = "頻率法 熱號 Frequency"
    kind = "統計"
    desc = "長週期熱號策略（賭徒謬誤）：用歷史頻率當機率。"

    def _next_proba(self, counts: np.ndarray) -> np.ndarray:
        return (counts + 1.0) / (counts.sum() + 49)

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        nums = parse_numbers(df)
        if len(nums) < 2:
            return _not_enough(self.name, self.kind, self.desc, 2, len(nums))

        counts = np.zeros(49, dtype=float)
        losses = []
        last_proba = None
        for t in range(len(nums)):
            if t > 0:
                counts += np.bincount(nums[t - 1], minlength=50)[:49]
            # 因果：第 t 期只用 t-1 及更早的計數
            proba = self._next_proba(counts)
            winning = nums[t]
            losses.append(-np.mean(np.log(proba[winning - 1] + 1e-12)))
            last_proba = proba

        mean_loss = float(np.mean(losses))
        # 下一期嘅 top-6（按概率由高到低）
        top = np.argsort(last_proba)[::-1][:6].tolist() if last_proba is not None else None
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "predicted_numbers": [int(n) + 1 for n in top] if top is not None else None,
        }


class MarkovTrendModel(Predictor):
    """
    馬爾可夫鏈 趨勢模型。

    為每個號碼建立一個 2 狀態（存在 / 不存在）的一階馬爾可夫鏈，
    估計「由上一期的狀態，transitions 到本期存在的機率」：
      * 若上期存在 → P(繼續存在)
      * 若上期不存在 → P(轉為存在)

    即捕捉「熱號持續、冷號翻生」的轉移趨勢，再歸一化成機率分佈。
    """

    name = "馬爾可夫鏈 趨勢 Markov"
    kind = "統計"
    desc = "每個號碼的 2 狀態轉移：捕捉存在/不存在的持續與翻生趨勢。"

    # 狀態約定（必須同 _add_transition 一致）：
    #   T[0, :] = 「上期**不存在**」→ [P(仍不存在), P(轉為存在)]
    #   T[1, :] = 「上期**存在**」  → [P(轉為不存在), P(繼續存在)]

    @classmethod
    def _proba_from(cls, T: np.ndarray, state: np.ndarray) -> np.ndarray:
        """依「上一期嘅狀態」(state = 上一期號碼) 算下期各號碼存在嘅機率分佈。

        上期存在嘅號碼 → P(繼續存在)；上期不存在嘅號碼 → P(轉為存在)。
        兩個量都歸一化成總和 = 1 嘅分佈（49 類，可與 ln(49) 基線直接比較）。
        """
        present = set(int(x) for x in state.tolist())
        pp = np.zeros(49)
        for i in range(1, 50):
            a = 1 if i in present else 0
            row_sum = T[a].sum()
            pp[i - 1] = T[a, 1] / row_sum if row_sum > 0 else 0.5
        total = pp.sum()
        return pp / total if total > 0 else np.full(49, 1.0 / 49)

    @classmethod
    def _add_transition(cls, T: np.ndarray, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """把 (prev -> cur) 嘅轉移計入矩陣（每個號碼獨立 2 狀態鏈）。

        ⚠️ 行約定必須同 `_proba_from` 一致：行 1 = 上期存在，行 0 = 上期不存在。
        舊版本 `_add_transition` 用 `int(i not in prev_set)`（行 1 = 上期**不存在**），
        同 `_proba_from` 嘅 `int(i in present)`（行 1 = 上期存在）**正好相反** →
        轉移矩陣嘅兩行被互換解讀：「熱號持續」被讀成「熱號死亡」，模型會預測
        「上期冇出現嘅號碼」，top-6 全部係上一期**冇**攪到嘅號碼（語義完全反）。
        在真實隨機數據上兩個約定係對稱嘅（都 ≈ ln 49 基線），所以佢喺真實數據
        上隱藏咗；但喺任何有趨勢嘅數據上都會令預測方向反曬。
        """
        prev_set = set(int(x) for x in prev.tolist())
        cur_set = set(int(x) for x in cur.tolist())
        for i in cur_set:                       # 本期存在
            T[int(i in prev_set), 1] += 1.0    # 存在→存在 / 不存在→存在
        for i in set(range(1, 50)) - cur_set:  # 本期不存在
            T[int(i in prev_set), 0] += 1.0    # 存在→不存在 / 不存在→不存在
        return T

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        nums = parse_numbers(df)
        if len(nums) < 2:
            return _not_enough(self.name, self.kind, self.desc, 2, len(nums))

        # --- 下一期預測（用全部歷史，狀態 = 最後一期）：儀表板顯示用 ---
        T_full = np.full((2, 2), 1.0)  # 加一平滑
        for prev, cur in zip(nums, nums[1:]):
            self._add_transition(T_full, prev, cur)
        display_proba = self._proba_from(T_full, nums[-1])
        top = np.argsort(display_proba)[::-1][:6].tolist()

        # --- rolling-origin 評估（增量更新轉移計數，避免 O(n²））---
        T = np.full((2, 2), 1.0)
        losses = []
        for t in range(1, len(nums)):
            # ⚠️ 預測第 t 期時，「當前狀態」必須係第 t-1 期嘅號碼。
            # 舊版本永遠用 nums[-1]（最後一期）作為狀態，等於用未來嘅狀態
            # 預測過去 → 泄漏，log-loss 完全唔可信。
            p = self._proba_from(T, nums[t - 1])
            losses.append(-np.mean(np.log(p[nums[t] - 1] + 1e-12)))
            self._add_transition(T, nums[t - 1], nums[t])

        mean_loss = float(np.mean(losses))
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "predicted_numbers": [int(n) + 1 for n in top],  # 按概率順序（下一期）
        }
