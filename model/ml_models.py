"""
sklearn 系列分類模型
==================

共用同一套「多分類 + 時間序列交叉驗證」評估骨架，只替換 estimator。
涵蓋：邏輯回歸（線性）、隨機森林（集成樹）、高斯樸素貝葉斯、神經網絡 MLP。

⚠️ 六合彩係獨立隨機事件，這些模型都唔會真的贏過均勻隨機。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from .base import Predictor, parse_numbers, multiclass_samples, uniform_baseline, window_features


class SklearnMulticlassModel(Predictor):
    """
    共用骨架：用 `multiclass_samples` 做多分類，
    以時間順序切 70%/30% 做訓練／評估，返回平均 log-loss。

    為咗令 Web 端點夠快（全資料約 26000 樣本、490 維，RF/MLP 會好慢），
    訓練時會固定抽樣到 `_max_train` 個樣本（可重複、公平）。
    """

    estimator_name: str = "model"
    #: 每個 fold 訓練時最多用多少樣本（預設全用）
    _max_train: int = 10_000
    #: 時間序列交叉驗證的 fold 數（折數越少越快；1 折 = 前 70% 訓練、後 30% 評估）
    _n_folds: int = 1

    def _make(self):
        """回傳一個新的 estimator 實例。"""
        raise NotImplementedError

    def _fit_predict(self, X, y, Xtr, ytr, Xte):
        clf = self._make()
        clf.fit(Xtr, ytr)
        self._last_clf = clf  # 保留最後訓練好嘅 classifier，供 predict_numbers 用
        classes = clf.classes_
        proba = clf.predict_proba(Xte)  # 一次過算全部類別，唔會逐類重複算
        out = np.zeros((len(Xte), 49))
        for j, c in enumerate(classes):
            out[:, c - 1] = proba[:, j]
        return out

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        X, y = multiclass_samples(df, lookback)
        n = len(X)

        # 按時間順序切：前 70% 訓練、後 30% 評估（保留時間序列性質）。
        # 只用 1 折以加快速度；若 _n_folds>1 則用 TimeSeriesSplit 多折。
        train_frac = 0.7
        cut = int(n * train_frac)
        folds = [(list(range(0, cut)), list(range(cut, n)))]
        if self._n_folds > 1:
            folds = list(TimeSeriesSplit(n_splits=self._n_folds).split(X))

        losses = []
        rng = np.random.RandomState(0)
        for fold, (tr, te) in enumerate(folds):
            Xtr, ytr = X[tr], y[tr]
            m = len(Xtr)
            if m > self._max_train:
                idx = rng.choice(m, size=self._max_train, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            proba = self._fit_predict(X, y, Xtr, ytr, X[te])
            losses.append(log_loss(y[te], proba, labels=range(1, 50)))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "predicted_numbers": self._predict_top6(parse_numbers(df), lookback),
        }

    def _predict_top6(self, nums, lookback):
        """對最後一期嘅滯後頻率特徵做 predict，揀出機率最高嘅 6 個號碼。"""
        if not nums:
            return None
        clf = getattr(self, "_last_clf", None)
        if clf is None:
            return None
        x = window_features(nums, len(nums) - 1, lookback).reshape(1, -1)
        classes = clf.classes_
        proba = clf.predict_proba(x)[0]
        out = np.zeros(49)
        for j, c in enumerate(classes):
            out[c - 1] = proba[j]
        top = np.argsort(out)[::-1][:6].tolist()
        return [int(n) for n in sorted(top)]


class LogisticRegressionModel(SklearnMulticlassModel):
    name = "邏輯回歸 Logistic Regression"
    kind = "sklearn"
    desc = "線性模型：特徵 × 權重的線性組合（基線級 sklearn 模型）。"
    estimator_name = "logistic"
    _max_train = 8000

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        return OneVsRestClassifier(
            LogisticRegression(max_iter=500, solver="liblinear", random_state=0)
        )


class RandomForestModel(SklearnMulticlassModel):
    name = "隨機森林 Random Forest"
    kind = "sklearn"
    desc = "多棵決策樹集成，能捕捉非線性但容易對噪聲過擬合。"
    estimator_name = "rf"
    _max_train = 6000

    def _make(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=0,
            n_jobs=-1,
        )


class GaussianNBModel(SklearnMulticlassModel):
    name = "高斯樸素貝葉斯"
    kind = "sklearn"
    desc = "假設每個號碼的頻率呈常態分佈，用貝葉斯定理後驗推估。"
    estimator_name = "nb"
    _max_train = 8000

    def _make(self):
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()


class MLPModel(SklearnMulticlassModel):
    name = "神經網絡 MLP"
    kind = "深度學習"
    desc = "多層感知機，學習滯後頻率特徵到號碼的非線性映射。"
    estimator_name = "mlp"
    _max_train = 3000

    def _make(self):
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=0)
