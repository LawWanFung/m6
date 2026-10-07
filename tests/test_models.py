"""模型層迴歸測試（全部離線，無網絡）。

覆蓋的迴歸（對應 docs/BUG_TODO.md）：
  1. draw_folds 按「期」切分 → train/test 絕不共享同一期（特徵向量零重疊）
  2. max_train_draws / max_test_draws 按整期截斷，絕不劈開同一期
  3. predicted_numbers 預測「下一期」(t = len(nums))，不是最後一期已知攪珠
  4. Markov：評估狀態必須因果（t-1 期）；轉移矩陣行約定一致
     （舊版 _add_transition 行約定相反 → 預測 sticky 趨勢的「補集」）
  5. _proba_matrix 對未覆蓋類別補 1e-6 並重新歸一（行和 = 1，基線可比）
  6. 數據不足 → 返回 error 條目（不拋異常，/api/models 不會 500）
  7. 校準迴歸：隨機數據上各模型 log-loss 應在基線附近
     （舊版 LR≈33 / GNB≈26 會直接掛）
  8. multiclass_samples 記憶體預算：超預算自動截最近 N 期（唔爆內存）
  9. _predict_top6 拒用「其他模型」訓練出嚟嘅 _last_clf（owner 校驗）
"""
from __future__ import annotations

import csv

import numpy as np

import analyze.frequency as freqmod
from model.base import (
    draw_folds,
    multiclass_samples,
    uniform_baseline,
    window_features,
)
from model.ml_models import _proba_matrix
from model.registry import run_all
from model.stat_models import MarkovTrendModel


def _random_csv(tmp_path, n: int = 200, seed: int = 7, name: str = "random.csv"):
    """確定性偽隨機主表（DD/MM/YYYY 日期、6 個不重複號碼 1-49）。"""
    rng = np.random.default_rng(seed)
    p = tmp_path / name
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["draw_id", "date", "numbers", "extra_ball"])
        for i in range(n):
            nums = sorted(rng.choice(np.arange(1, 50), size=6, replace=False))
            others = [x for x in range(1, 50) if x not in nums]
            extra = int(others[int(rng.integers(len(others)))])
            day = (i % 28) + 1
            month = (i // 28) % 12 + 1
            w.writerow([f"D{i:04d}", f"{day:02d}/{month:02d}/2020",
                        ",".join(map(str, nums)), extra])
    return p


# ───────────────────────── 1. 按「期」切分，無泄漏 ─────────────────────────


def test_draw_folds_disjoint_and_no_vector_leak(tmp_path):
    df = freqmod.load(_random_csv(tmp_path, 200))
    X, y, draw_idx = multiclass_samples(df, 10)
    assert X is not None and len(X) > 0
    folds = draw_folds(draw_idx, n_folds=3, train_frac=0.7)
    assert folds
    for tr, te in folds:
        assert len(tr) > 0 and len(te) > 0
        # 同一期絕不能同時在兩邊
        assert set(draw_idx[tr]).isdisjoint(draw_idx[te])
        # 特徵向量層面零重疊（同一期 6 條樣本共用同一向量）
        tr_vecs = {tuple(np.round(X[i], 6)) for i in tr}
        te_vecs = {tuple(np.round(X[i], 6)) for i in te}
        assert tr_vecs.isdisjoint(te_vecs)


def test_max_draws_limits_whole_draws(tmp_path):
    df = freqmod.load(_random_csv(tmp_path, 200))
    X, y, draw_idx = multiclass_samples(df, 10)
    tr, te = draw_folds(draw_idx, n_folds=1, max_train_draws=50, max_test_draws=20)[0]
    assert len(set(draw_idx[tr])) == 50
    assert len(set(draw_idx[te])) == 20
    assert set(draw_idx[tr]).isdisjoint(draw_idx[te])


def test_draw_folds_empty_when_not_enough_draws(tmp_path):
    df = freqmod.load(_random_csv(tmp_path, 1))
    assert draw_folds(np.array([0]), n_folds=1) == []


# ───────────────────────── 2. 預測目標 = 下一期 ─────────────────────────


def test_window_for_next_draw_uses_latest_known_draw():
    # 12 期循環：第 i 期 = 組 (i % 8) → 號碼 (i%8)*6+1 .. +6
    nums = [np.array([(i % 8) * 6 + j + 1 for j in range(6)]) for i in range(12)]
    lb = 3
    x = window_features(nums, len(nums), lb)  # t = len → 尚未發生的「下一期」
    last = np.bincount(nums[-1], minlength=50)[1:50]
    assert np.array_equal(x[:49], last), "最新一格必須是最後一期（已知）"
    prev = np.bincount(nums[-2], minlength=50)[1:50]
    assert not np.array_equal(x[:49], prev), "舊版本 t=len-1 會把倒數第二期當最新"


# ───────────────────────── 3. Markov 因果 + 行約定 ─────────────────────────


def test_markov_causal_evaluation_and_sticky_prediction(make_csv):
    """80 期 sticky [1..6] → 20 期 [44..49]。

    正確的因果實現（狀態 = t-1）應學到「存在→繼續存在」趨勢，log-loss 遠低基線；
    預測必須跟着最後一期 [44..49]（sticky），而不是它的補集。
    """
    rows = [[f"D{i:03d}", "01/01/2020", "1,2,3,4,5,6", "7"] for i in range(80)] + \
           [[f"E{i:03d}", "01/01/2021", "44,45,46,47,48,49", "1"] for i in range(20)]
    df = freqmod.load(make_csv(rows))
    r = MarkovTrendModel().evaluate(df, 1)
    assert r["mean_log_loss"] is not None
    assert r["mean_log_loss"] < 3.0, (
        f"因果評估應學到 sticky 趨勢（ll<3），現得 {r['mean_log_loss']:.3f}"
        "（舊版行約定反了會是 ~10.6，舊版泄漏狀態會 > 基線）"
    )
    assert r["can_predict"] is True
    assert set(r["predicted_numbers"]) == {44, 45, 46, 47, 48, 49}, (
        "預測必須反映最後一期的 sticky 趨勢；補集 = 行約定反了"
    )


def test_markov_real_data_stays_at_baseline():
    """真實（隨機）數據上兩個行約定不可區分 → ll 應釘在基線附近。"""
    df = freqmod.load("data/mark6_history.csv")
    r = MarkovTrendModel().evaluate(df, 10)
    assert abs(r["mean_log_loss"] - uniform_baseline()) < 0.15


# ───────────────────────── 4. 缺失類別補全 ─────────────────────────


def test_proba_matrix_pads_missing_classes():
    class _Fake:
        classes_ = np.array([1, 5])

        def predict_proba(self, X):
            n = len(X) if hasattr(X, "__len__") else 1
            return np.tile(np.array([0.3, 0.7]), (n, 1))

    M, n_missing = _proba_matrix(_Fake(), np.zeros((10, 4)), n_numbers=49)
    assert M.shape == (10, 49)
    assert n_missing == 47
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-9)
    assert np.isfinite(M).all()
    assert abs(M[0, 0] - 0.3) < 0.01   # 已歸一化，近似原概率
    assert abs(M[0, 4] - 0.7) < 0.01


def test_proba_matrix_none_classes_falls_back_to_uniform():
    class _NoClasses:
        def predict_proba(self, X):
            return np.ones((len(X), 1))

    M, n_missing = _proba_matrix(_NoClasses(), np.zeros((5, 2)), n_numbers=49)
    np.testing.assert_allclose(M, 1.0 / 49, atol=1e-12)
    assert n_missing == 49


# ───────────────────────── 5. 數據不足 → error 條目 ─────────────────────────


def test_insufficient_data_returns_error_entries(tmp_path, make_csv):
    rows = [[f"D{i}", "01/01/2020", "1,2,3,4,5,6", "7"] for i in range(3)]
    df = freqmod.load(make_csv(rows))
    results = run_all(df, lookback=10)   # 3 期 < lookback → ML 模型無法建模
    assert len(results) == 7
    for r in results:
        assert "mean_log_loss" in r and "can_predict" in r
    errs = [r for r in results if r.get("error")]
    assert len(errs) == 4, "4 個 sklearn 模型應報 error（統計模型 3 期仍夠）"
    for r in errs:
        assert r["mean_log_loss"] is None
        assert r["can_predict"] is False
        assert r["predicted_numbers"] is None
    ok = [r for r in results if not r.get("error")]
    assert all(r["mean_log_loss"] is not None and np.isfinite(r["mean_log_loss"])
               for r in ok)


# ───────────────────────── 6. 校準迴歸 ─────────────────────────


def test_calibration_within_baseline_band(tmp_path):
    """隨機數據：任何模型都不應顯著「贏」或「輸」基線太多。

    舊版無正則 LR（ll≈33）/ 未去相關 GNB（ll≈26）會直接掛在這裡。
    """
    df = freqmod.load(_random_csv(tmp_path, 300, seed=11))
    results = run_all(df, lookback=10)
    assert len(results) == 7
    for r in results:
        ll = r["mean_log_loss"]
        assert ll is not None, f"{r['name']} 應有結果，error={r.get('error')}"
        assert 2.5 < ll < 7.0, (
            f"{r['name']} ll={ll:.3f} 超出合理區間 (2.5, 7.0) —— 校準 artifact"
        )


# ───────────── 8. 記憶體預算（R2-4）─────────────


def test_multiclass_samples_respects_byte_budget(tmp_path):
    """lookback 調大令特徵矩陣超預算 → 自動截最近 N 期，唔爆內存。"""
    df = freqmod.load(_random_csv(tmp_path, 120))
    X, y, draw_idx = multiclass_samples(df, 10, byte_budget=None)
    full_rows = len(X)
    assert full_rows == (120 - 10) * 6

    # 極小預算（~100 KB，正常會要幾 MB）→ 必觸發截斷
    X2, y2, di2 = multiclass_samples(df, 10, byte_budget=100_000)
    assert X2 is not None and len(X2) < full_rows
    # 截斷只丟最舊：draw_idx 全部指向原序列後段
    assert min(int(d) for d in di2) > 0
    assert max(int(d) for d in di2) == 119


# ───────────── 9. _last_clf owner 校驗（R2-23）─────────────


def test_predict_top6_rejects_foreign_last_clf(make_csv):
    """_last_clf 若係其他模型訓練出嚟（owner 唔匹配）→ 當「無預測」，唔借權重。"""
    from model.ml_models import LogisticRegressionModel

    rows = [[f"D{i}", "01/01/2020", "1,2,3,4,5,6", "7"] for i in range(15)]
    df = freqmod.load(make_csv(rows))
    m = LogisticRegressionModel()

    class _ForeignClf:
        _owner = "其他模型"
        classes_ = np.arange(1, 50)

        def predict_proba(self, X):
            return np.full((len(X), 49), 1 / 49)

    m._last_clf = _ForeignClf()
    assert m._predict_top6(
        [np.array([1, 2, 3, 4, 5, 6]) for _ in range(15)], 10) is None

    # 自己訓練出嚟嘅 → 正常出預測
    m._last_clf = None
    X, y, _ = multiclass_samples(df, 10)
    m._fit_predict(X, y, X[:30])
    got = m._predict_top6(
        [np.array([1, 2, 3, 4, 5, 6]) for _ in range(15)], 10)
    assert got is not None and len(got) == 6
