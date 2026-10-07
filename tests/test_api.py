"""FastAPI 端點回歸測試（TestClient，離線）。

覆蓋的迴歸（對應 docs/BUG_TODO.md）：
  1. 數據源缺失 → 503（絕不用合成樣本頂替真實數據）
  2. /api/models 數據不足（n <= lookback）→ 400，不是 500
  3. lookback 越界 → 422（Query 校驗）
  4. /api/models 與 /api/model 的緩存 key 命名空間隔離
     （舊版共用 key → /api/models 會錯誤回傳 dict）
  5. FETCH_TOKEN：未帶／帶錯 token → 403；**未設** token → 503（端點停用）
  6. 抓取失敗（子程序 exit 1）→ 任務 status = error，不是 success
  7. 單飛：已有任務進行中 → busy，不重複抓取
"""
from __future__ import annotations

import subprocess
import time

import pytest
from fastapi.testclient import TestClient

import app.main as app_main


@pytest.fixture(scope="module")
def client():
    return TestClient(app_main.app)


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    app_main._fetch_tasks.clear()
    if app_main._fetch_lock.locked():
        app_main._fetch_lock.release()
    app_main._model_cache.clear()
    app_main._analysis_cache.clear()


def _fake_run_fail(args, **kwargs):
    return subprocess.CompletedProcess(args=args, returncode=1,
                                       stdout="", stderr="boom (test)")


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "六合彩" in r.text


def test_health_reports_real_source(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data"]["data_source"] == "real"
    assert body["data"]["n_draws"] > 4000
    assert body["fetch_busy"] is False


def test_analysis_matches_health(client):
    a = client.get("/api/analysis").json()
    h = client.get("/api/health").json()["data"]
    assert a["n_draws"] == h["n_draws"]
    assert a.get("extra_chi2") is not None
    # 緩存：第二次調用不應重算（同一對象）
    a2 = client.get("/api/analysis").json()
    assert a == a2


def test_models_full_run(client):
    r = client.get("/api/models?lookback=10")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["models"], list) and len(body["models"]) == 7
    for m in body["models"]:
        assert "mean_log_loss" in m
        assert "uniform_baseline_log_loss" in m
        assert "predicted_numbers" in m
    assert abs(body["models"][0]["mean_log_loss"] - 3.8918) < 0.001  # 均勻基線


def test_models_lookback_validation(client):
    assert client.get("/api/models?lookback=9999").status_code == 422
    assert client.get("/api/models?lookback=0").status_code == 422


def test_models_small_data_returns_400(client, monkeypatch, make_csv):
    rows = [[f"D{i}", "01/01/2020", "1,2,3,4,5,6", "7"] for i in range(5)]
    small = make_csv(rows)
    monkeypatch.setattr(app_main, "HISTORY_CSV", small)
    app_main._analysis_cache.clear()
    app_main._model_cache.clear()
    r = client.get("/api/models?lookback=10")
    assert r.status_code == 400
    assert "lookback" in r.json()["detail"]


def test_missing_real_data_returns_503_never_sample(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_main, "HISTORY_CSV", tmp_path / "does_not_exist.csv")
    app_main._analysis_cache.clear()
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["data"]["data_source"] == "unavailable"
    r = client.get("/api/analysis")
    assert r.status_code == 503, "真實數據缺失必須 503，絕不用合成樣本"
    assert client.get("/api/models?lookback=10").status_code == 503


def test_models_and_single_model_cache_coexist(client):
    """緩存 key 命名空間迴歸：兩個端點先後調用互不污染。"""
    r1 = client.get("/api/models?lookback=10")
    assert isinstance(r1.json()["models"], list)
    r2 = client.get("/api/model?lookback=10")
    assert r2.status_code == 200 and isinstance(r2.json(), dict)
    r3 = client.get("/api/models?lookback=10")   # 舊版這裡會回傳 dict
    assert isinstance(r3.json()["models"], list)
    assert len(r3.json()["models"]) == 7


def test_fetch_requires_token(client, monkeypatch):
    monkeypatch.setenv("FETCH_TOKEN", "s3cret")
    monkeypatch.setattr(app_main.subprocess, "run", _fake_run_fail)

    assert client.post("/api/fetch").status_code == 403

    r = client.post("/api/fetch", headers={"X-Fetch-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["status"] == "fetching"

    assert client.post("/api/fetch", headers={"X-Fetch-Token": "wrong"}).status_code == 403


def test_fetch_unset_token_returns_503(client, monkeypatch):
    """R2-2 迴歸：未設 FETCH_TOKEN 時端點一律停用（回 503）。

    公開部署若忘記設 token，舊版（`required and ...`）會直接放行 →
    任何訪客都能觸發「抓外部網站 + 重寫主表」。
    """
    monkeypatch.delenv("FETCH_TOKEN", raising=False)
    monkeypatch.setattr(app_main.subprocess, "run", _fake_run_fail)

    r = client.post("/api/fetch")
    assert r.status_code == 503
    assert "FETCH_TOKEN" in r.json()["detail"]
    # 帶了 token 也沒用（根本冇人要求）
    assert client.post("/api/fetch", headers={"X-Fetch-Token": "anything"}).status_code == 503


def test_fetch_failure_propagates_as_error(client, monkeypatch):
    """P0 迴歸：子程序非零 exit → 任務 status=error（舊版誤報 success）。"""
    monkeypatch.setenv("FETCH_TOKEN", "test-tok")
    monkeypatch.setattr(app_main.subprocess, "run", _fake_run_fail)

    r = client.post("/api/fetch", headers={"X-Fetch-Token": "test-tok"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    st = {}
    for _ in range(100):
        st = client.get("/api/fetch_status", params={"task_id": task_id}).json()
        if st.get("status") in ("success", "error"):
            break
        time.sleep(0.05)
    assert st["status"] == "error"
    assert "抓取失敗" in st.get("error", "")
    assert client.get("/api/health").json()["fetch_busy"] is False


def test_fetch_single_flight(client, monkeypatch):
    monkeypatch.setenv("FETCH_TOKEN", "test-tok")
    monkeypatch.setattr(app_main.subprocess, "run", _fake_run_fail)
    app_main._fetch_lock.acquire()
    try:
        r = client.post("/api/fetch", headers={"X-Fetch-Token": "test-tok"})
        assert r.status_code == 200
        assert r.json()["status"] == "busy"
    finally:
        if app_main._fetch_lock.locked():
            app_main._fetch_lock.release()
