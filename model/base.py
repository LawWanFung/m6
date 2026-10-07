"""
建模基底（共用特徵工程與評估框架）
=================================

⚠️ 六合彩係**獨立隨機事件**，任何基於歷史號碼的模型都**無法真正預測**下一期。
本模組的目標係「示範流程」並「證實無法預測」：每個模型輸出一個 1–49 嘅機率分佈，
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
#: multiclass 特徵矩陣嘅記憶體預算（位元組）。
# ⚠️ 全量歷史 × 大 lookback 會爆內存（5000 期 × lookback 60 ≈ 768MB）；
# 超預算時自動截成「最近 N 期」。預設 lookback（10）遠低於預算，結果唔受影響。
_MULTICLASS_SAMPLE_BYTES = 256 * 1024 * 1024


def parse_numbers(df: pd.DataFrame) -> List[np.ndarray]:
    """將 `numbers` 欄解析成每期一個的 int numpy 陣列。

    相容兩種格式：
      * "1,8,13,24,35,43"   （sample CSV / 原始 CSV）
      * "[1, 8, 13, 24, 35, 43]"  （主表 CSV，含方括號）

    ⚠️ 唔開獎／損壞嘅行（空 numbers、"nan"、少於 6 個號碼）會被**跳過**，
    唔會拋 ValueError — 否則一條壞行就會令整個建模 pipeline 崩潰（/api/model 500）。
    """
    out = []
    for v in df["numbers"]:
        s = str(v).strip().strip("[]")
        if not s or s.lower() == "nan":
            continue
        parts = [x.strip() for x in s.split(",") if x.strip()]
        if len(parts) != 6 or not all(p.isdigit() for p in parts):
            continue
        out.append(np.array(sorted(int(p) for p in parts), dtype=int))
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


def multiclass_samples(df: pd.DataFrame, lookback: int,
                       byte_budget: int | None = _MULTICLASS_SAMPLE_BYTES):
    """
    將「預測下期各號碼機率」轉成多分類問題：

    對每個 `t >= lookback`，以第 `t` 期嘅滯後頻率特徵為輸入，
    第 `t` 期嘅每個主號為一個正樣本（label = 該號碼）。
    即係每個 ``(期, 號碼)`` 都是一條樣本，訓練集好大。

    返回 `(X, y, draw_idx)`；無樣本時回 `(None, None, None)`。
    `draw_idx[i]` 係樣本 i 所屬嘅「攪珠期索引」t。

    ⚠️ 必須用 `draw_idx` 按「期」切訓練／評估集：同一期嘅 6 條樣本共用同一
    特徵向量。按樣本索引切（例如 `n*0.7` 或 `TimeSeriesSplit.split(X)`）會在
    切點處把同一期嘅樣本分成兩邊 → 同一特徵向量同時出現在訓練同評估集，
    模型記得住 → 泄漏、log-loss 唔公平。

    記憶體保護：若特徵矩陣 byte 量超 `byte_budget`，只保留最近 N 期
    （見 `_MULTICLASS_SAMPLE_BYTES`）；預設 lookback 唔會觸發。
    """
    nums = parse_numbers(df)
    offset = 0  # 截斷後補回原索引（draw_idx 必須保持絕對位置）
    if byte_budget and len(nums) > lookback:
        dim = lookback * 49  # window_features 輸出維度
        if len(nums) * dim * 8 > byte_budget:
            keep = byte_budget // (6 * 8 * dim) + lookback
            if keep < len(nums):
                offset = len(nums) - keep
                print(
                    f"[!] multiclass_samples: 特徵矩陣超記憶體預算"
                    f"（{byte_budget // 2**20} MB），只保留最近 {keep} 期"
                    f"（原 {len(nums)} 期）；lookback 調大時請留意。"
                )
                nums = nums[-keep:]
    X, y, draw_idx = [], [], []
    for t in range(lookback, len(nums)):
        x = window_features(nums, t, lookback)
        for n in nums[t]:
            X.append(x)
            y.append(n)
            draw_idx.append(t + offset)
    if not X:
        return None, None, None
    return np.vstack(X), np.array(y), np.array(draw_idx)


def pick_draws(draws: list[int], max_draws: int | None) -> list[int]:
    """取最近的 N 期（保持時間順序），唔拆同一期嘅 6 條樣本。"""
    if not max_draws or len(draws) <= max_draws:
        return list(draws)
    return list(draws[-max_draws:])


def draw_folds(draw_idx, n_folds: int = 1, train_frac: float = 0.7,
               max_train_draws: int | None = None, max_test_draws: int | None = None):
    """按「攪珠期」切 fold，回傳 `[(train_sample_idx, test_sample_idx), ...]`。

    - 1 fold：前 `train_frac` 的期做訓練，其餘做評估（保留時間順序）。
    - `max_train_draws` / `max_test_draws` 限制每個 fold 的訓練／評估**期數**
      （不是樣本數），取最近的期。這既避免同一期嘅特徵向量跨過訓練／評估集
      （泄漏），亦令 Web 端點運行時可控：49 類 × 490 維 × 上萬樣本會令
      /api/models 每請求 14-400 秒。
    """
    draws = sorted(set(int(d) for d in draw_idx))
    if len(draws) < 2:
        return []

    by_draw: dict[int, list[int]] = {}
    for i, d in enumerate(draw_idx):
        by_draw.setdefault(int(d), []).append(i)

    def fold(tr_draws, te_draws):
        tr_draws = pick_draws(tr_draws, max_train_draws)
        te_draws = pick_draws(te_draws, max_test_draws)
        tr = [i for d in tr_draws for i in by_draw[d]]
        te = [i for d in te_draws for i in by_draw[d]]
        return tr, te

    if n_folds <= 1:
        cut = max(1, int(len(draws) * train_frac))
        if cut >= len(draws):
            cut = len(draws) - 1
        return [fold(draws[:cut], draws[cut:])] if cut > 0 else []

    per = max(1, len(draws) // (n_folds + 1))
    out = []
    for k in range(n_folds):
        tr, te = draws[:(k + 1) * per], draws[(k + 1) * per:(k + 2) * per]
        if tr and te:
            out.append(fold(tr, te))
    return out


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
