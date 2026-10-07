"""
sklearn 系列分類模型
==================

共用同一套「多分類 + 按攪珠期切分」評估骨架，只替換 estimator。
涵蓋：邏輯迴歸（線性）、隨機森林（集成樹）、高斯樸素貝葉斯、神經網絡 MLP。

⚠️ 六合彩係獨立隨機事件，這些模型都唔會真的贏過均勻隨機。

評估要點（唔可以妥協）：
  1. **按「期」切分**，唔按樣本切。同一期嘅 6 條樣本共用同一特徵向量；按樣本
     切會在切點處把同一期嘅樣本分成兩邊 → 同一向量同時出現在訓練同評估集（泄漏）。
  2. `predict_proba` 每 fold 只算一次（唔可以喺 49 個類別循環裡重算）。
  3. 訓練集未覆蓋嘅號碼會有一整列 0；`log_loss` 會把 0 clip 到 EPS，令 loss
     變成無意義的大值。所以填滿 EPS 並重新歸一化，同時回傳 `class_coverage`。
  4. `mean_p_correct` ≈ 1/49 (=0.0204) 代表「模型與隨機無異」；明顯低於呢個值
     代表模型 confidently 錯（degenerate 概率輸出），唔係「有信號」。
  5. `predicted_numbers` 係**下一期**的 top-6（按概率順序），唔係最後一期
     已知攪珠的「重算」。
  6. `_max_train_draws` / `_max_test_draws` 限制評估規模（以「期」為單位），
      否則 /api/models 每次請求要 14-400 秒。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .base import (
    Predictor,
    parse_numbers,
    multiclass_samples,
    uniform_baseline,
    window_features,
    draw_folds,
)


def _classes_of(clf):
    """Pipeline 冇 `classes_`；取最後一步的 estimator。"""
    if hasattr(clf, "classes_"):
        return clf.classes_
    if hasattr(clf, "steps"):
        return getattr(clf.steps[-1][1], "classes_", None)
    return None


def _proba_matrix(clf, X, n_numbers: int = 49) -> tuple[np.ndarray, int]:
    """一次過算 predict_proba，返回 (proba, 未覆蓋類別數)。

    未覆蓋嘅類別（訓練集從未出現該號碼）會留 0 列；填滿 1e-6 後歸一化，
    否則 `log_loss(..., labels=range(1,50))` 會把 0 clip 成 eps，得到無意義
    的大值，且同 ln(49) 基線不可比。
    """
    classes = _classes_of(clf)
    if classes is None:
        return np.full((len(X), n_numbers), 1.0 / n_numbers), n_numbers

    proba = clf.predict_proba(X)
    out = np.zeros((len(X), n_numbers))
    seen = set()
    for j, c in enumerate(classes):
        c = int(c)
        if 1 <= c <= n_numbers:
            out[:, c - 1] = proba[:, j]
            seen.add(c)

    missing = [c for c in range(1, n_numbers + 1) if c not in seen]
    if missing:
        for c in missing:
            out[:, c - 1] = 1e-6
        out = out / out.sum(axis=1, keepdims=True)
    return out, len(missing)


class SklearnMulticlassModel(Predictor):
    """
    共用骨架：用 `multiclass_samples` 做多分類，按「攪珠期」時間順序切
    訓練／評估（保留時間序列性質），返回平均 log-loss。
    """

    estimator_name: str = "model"
    #: 每個 fold 最多用多少「期」做訓練（None = 全用）
    _max_train_draws: int | None = None
    #: 每個 fold 最多用多少「期」做評估（取最近的期）
    _max_test_draws: int | None = None
    #: 時間序列交叉驗證的 fold 數（1 折 = 前 70% 訓練、後 30% 評估）
    _n_folds: int = 1

    def _make(self):
        """回傳一個新的 estimator 實例（每次 evaluate 都新造，唔共享狀態）。"""
        raise NotImplementedError

    def _fit_predict(self, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray):
        clf = self._make()
        clf.fit(Xtr, ytr)
        clf._owner = self.name  # 標籤「呢個分類器係邊個模型訓練」，供 _predict_top6 校驗
        self._last_clf = clf  # 供 _predict_top6 用（下一期預測）
        return _proba_matrix(clf, Xte)

    def evaluate(self, df: pd.DataFrame, lookback: int) -> dict:
        nums = parse_numbers(df)
        X, y, draw_idx = multiclass_samples(df, lookback)

        if X is None or len(X) == 0:
            return {
                "name": self.name,
                "kind": self.kind,
                "description": self.desc,
                "mean_log_loss": None,
                "uniform_baseline_log_loss": uniform_baseline(),
                "can_predict": False,
                "predicted_numbers": None,
                "error": f"數據不足：需要多於 lookback={lookback} 期（現得 {len(nums)} 期）",
            }

        folds = draw_folds(draw_idx, n_folds=self._n_folds, train_frac=0.7,
                           max_train_draws=self._max_train_draws,
                           max_test_draws=self._max_test_draws)
        if not folds:
            return {
                "name": self.name,
                "kind": self.kind,
                "description": self.desc,
                "mean_log_loss": None,
                "uniform_baseline_log_loss": uniform_baseline(),
                "can_predict": False,
                "predicted_numbers": None,
                "error": "數據不足：無法按攪珠期切出訓練/評估集",
            }

        losses: list[float] = []
        coverage: list[float] = []
        p_correct: list[float] = []
        n_test = 0
        for tr, te in folds:
            if not tr or not te:
                continue
            proba, n_missing = self._fit_predict(X[tr], y[tr], X[te])
            losses.append(float(log_loss(y[te], proba, labels=range(1, 50))))
            coverage.append((49 - n_missing) / 49)
            p_correct.append(float(np.mean(proba[np.arange(len(te)), y[te] - 1])))
            n_test += len(te)

        mean_loss = float(np.mean(losses))
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.desc,
            "mean_log_loss": mean_loss,
            "uniform_baseline_log_loss": uniform_baseline(),
            "can_predict": self.verdict(mean_loss),
            "class_coverage": round(float(np.mean(coverage)), 4),
            "mean_p_correct": round(float(np.mean(p_correct)), 6),
            "n_folds": len(folds),
            "n_test_samples": n_test,
            "predicted_numbers": self._predict_top6(nums, lookback),
        }

    def _predict_top6(self, nums, lookback: int):
        """**下一期**的 top-6（按概率由高到低）。

        ⚠️ 特徵窗口必須用 `t = len(nums)`（即「預測尚未發生的那一期」）。
        用 `len(nums) - 1` 會評估最後一期**已知**攪珠，等於「用答案算答案」，
        輸出的號碼會重疊最後一期實際號碼，UI 的「下一期預測」完全冇意義。
        """
        if not nums or len(nums) <= lookback:
            return None
        clf = getattr(self, "_last_clf", None)
        # ⚠️ 防止「借走」嘅狀態：`_last_clf` 若係**其他模型**訓練出嚟嘅分類器
        # （例如直接調 predict.run()、或未來重構時 registry 共用實例）就用
        # 另一個模型嘅權重出「下一期預測」—— owner 唔匹配時當「無預測」。
        if clf is None or getattr(clf, "_owner", self.name) != self.name:
            return None
        x = window_features(nums, len(nums), lookback).reshape(1, -1)
        proba, _ = _proba_matrix(clf, x)
        top = np.argsort(proba[0])[::-1][:6]
        return [int(n) + 1 for n in top]  # 保持概率順序，唔用 sorted() 打亂


class LogisticRegressionModel(SklearnMulticlassModel):
    name = "邏輯回歸 Logistic Regression"
    kind = "sklearn"
    desc = "線性模型（特徵已標準化）：基線級 sklearn 模型。"
    estimator_name = "logistic"
    _max_train_draws = 600
    _max_test_draws = 300

    def _make(self):
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ⚠️ 關鍵：必須帶 L2 正則（alpha=1.0，較強收縮）。
        # 舊版本 `penalty=None` = 純 MLE：490 維滯後 one-hot 特徵對純隨機標籤
        # 會「過擬合噪聲」→ 權重巨大 → 概率退化成接近 0/1 → 93% 的 test 樣本
        # 真號碼概率 < 1e-9 → log-loss 33-35，看起來像「模型很差」，其實係
        # 校準 artifact。帶 L2 後模型收縮回先驗，log-loss = 3.90 ≈ ln(49) 基線，
        # mean_p_correct = 0.0204 ≈ 1/49 → 誠實地報告「冇信號」（這才係真相）。
        # 舊版本：OneVsRestClassifier(LogisticRegression(liblinear, 500 iter)) 對
        # 490 維特徵要 400+ 秒；SGD 約 2-4 秒。
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="log_loss", penalty="l2", alpha=1.0,
                                  max_iter=200, random_state=0, tol=1e-4)),
        ])


class RandomForestModel(SklearnMulticlassModel):
    name = "隨機森林 Random Forest"
    kind = "sklearn"
    desc = "多棵決策樹集成，能捕捉非線性但容易對噪聲過擬合。"
    estimator_name = "rf"
    _max_train_draws = 1000
    _max_test_draws = 400

    def _make(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=0,
            n_jobs=-1,
        )


class GaussianNBModel(SklearnMulticlassModel):
    name = "高斯樸素貝葉斯"
    kind = "sklearn"
    desc = "假設每個號碼的頻率呈常態分佈，用貝葉斯定理後驗推估（已知會輸）。"
    estimator_name = "nb"
    _max_train_draws = 1000
    _max_test_draws = 300

    def _make(self):
        from sklearn.decomposition import PCA
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ⚠️ 關鍵：GNB 前必須 PCA 去相關。490 個滯後 one-hot 特徵強相關（每期
        # 恰好 6 個為 1），而 NB 假設特徵獨立 → 490 維的條件概率連乘會累積誤差，
        # 概率退化（68-72% 的 test 樣本真號碼概率 < 1e-9 → log-loss 25+）。
        # PCA 成分兩兩不相關，NB 的獨立性假設才成立：
        #   PCA(10) → ll 3.93，PCA(20) → ll 3.99，均 ≈ ln(49)=3.89 基線，
        #   mean_p_correct ≈ 1/49 → 誠實的「冇信號」結果。
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20, random_state=0)),
            ("clf", GaussianNB()),
        ])


class MLPModel(SklearnMulticlassModel):
    name = "神經網絡 MLP"
    kind = "深度學習"
    desc = "多層感知機（特徵已標準化），學習滯後頻率特徵到號碼的映射。"
    estimator_name = "mlp"
    _max_train_draws = 500
    _max_test_draws = 300

    def _make(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # 舊版本冇標準化且 max_iter=100 → 永遠唔會收斂，log-loss 9.15 係
        # 「未收斂」嘅 artifacts，唔係「模型冇信號」的證據。
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200, n_iter_no_change=10,
                early_stopping=True, random_state=0,
            )),
        ])
