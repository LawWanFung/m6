"""
六合彩數據儀表板 — FastAPI Web App
===================================

端點：
  GET  /                儀表板 HTML
  GET  /api/analysis    統計分析（緩存）
  GET  /api/model       單一模型（MLP）結果（緩存）
  GET  /api/models      全部模型比較 + 分析（緩存）
  GET  /api/health      健康檢查 + 數據來源信息
  POST /api/fetch       觸發後台抓取 + 整合（單飛、有超時；必須設 FETCH_TOKEN
                         並帶 X-Fetch-Token，未設 token 時統一回 503 停用）

設計要點（舊版本嘅缺陷，已修正）：
  1. 重 CPU 端點以前係 `async def` → 模型訓練阻塞 event loop，連 `/api/health`
     都要等 13 秒。現在用普通 `def`（FastAPI 跑在 threadpool）+ 結果緩存。
  2. `/api/fetch` 以前無論子程序成功/失敗都回傳 `success`，且無單飛保護 →
     網絡失敗會被誤報為「已更新但無新增」，而且併發請求會同時重寫主表。
     現在：同一時間只允許一個 fetch；子程序非零 exit → status = "error"。
  3. `data_source` 以前 hardcode "real" → 現在從實際加載的數據來源推導。
  4. 未傳 `uvicorn` import → `python app/main.py` 直接 NameError。已修。
"""

from __future__ import annotations

import hmac
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - 有些 stdout 不可 reconfigure
    pass

import pandas as pd

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

import analyze.frequency as freqmod
from model.predict import run as model_run
from model.registry import run_all

TEMPLATES_DIR = Path(__file__).parent / "templates"
SRC = BASE / "src"
#: 真實歷史主表（分析/模型唯一數據來源；已入 repo）
HISTORY_CSV = BASE / "data" / "mark6_history.csv"

# ───────────────────────── 緩存 / 併發控制 ─────────────────────────
_model_cache: dict[tuple[int, float | None], dict] = {}
_analysis_cache: dict[tuple[float, int], dict] = {}
_cache_lock = threading.Lock()
#: 分析緩存上限（key = CSV (mtime, size)；每次 fetch 後變一個新 key）
MAX_ANALYSIS_CACHE = 8
#: 重 CPU 計算串行化：避免 4 個併發請求各自跑 6 秒模型（互相搶 CPU）
heavy_lock = threading.Lock()

# ───────────────────────── fetch 任務登記 ─────────────────────────
_fetch_tasks: dict[str, dict] = {}
#: 單飛：同一時間只允許一個抓取任務（避免併發寫主表 / 重複抓取）
_fetch_lock = threading.Lock()
MAX_TASKS = 20
TASK_TTL = 3600.0          # 任務保留 1 小時，之後自動清理（避免內存無限增長）
FETCH_TIMEOUT = 900        # hkjc_fetch.py 最長運行時間（秒）
BUILD_TIMEOUT = 300        # build_history.py 最長運行時間（秒）

app = FastAPI(title="Mark Six 儀表板", version="1.1.0")


# ───────────────────────── 數據加載 ─────────────────────────

def _csv_stamp() -> tuple[float, int]:
    """用 (mtime, size) 作為緩存 key；數據變更後緩存自動失效。"""
    try:
        st = HISTORY_CSV.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def _load_df() -> pd.DataFrame:
    """加載**真實**歷史數據（data/mark6_history.csv）。

    明確唔傳 `allow_sample`：主表不存在/為空就 503，**絕對不會**退回合成樣本，
    以免儀表板顯示 demo 數據而令人誤以為係真實開獎結果。
    """
    try:
        df = freqmod.load(HISTORY_CSV)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"真實歷史數據不可用：{exc}\n"
                "請執行：python src/hkjc_fetch.py --from 1993-01-01 然後 "
                "python src/build_history.py --mode build"
            ),
        ) from exc

    if df.attrs.get("data_source") != "real":
        raise HTTPException(
            status_code=503,
            detail=f"數據來源不是真實 HKJC 數據（{df.attrs.get('data_source')}），已中止分析。",
        )
    return df


def _cap_analysis_cache() -> None:
    """必須喺 `_cache_lock` 內調用：分析緩存保留上限（淘汰最舊的一半）。"""
    if len(_analysis_cache) < MAX_ANALYSIS_CACHE:
        return
    for k in list(_analysis_cache)[: len(_analysis_cache) - MAX_ANALYSIS_CACHE // 2]:
        _analysis_cache.pop(k, None)


def _get_analysis() -> dict:
    stamp = _csv_stamp()
    with _cache_lock:
        if stamp in _analysis_cache:
            return _analysis_cache[stamp]
        _cap_analysis_cache()
    with heavy_lock:
        data = freqmod.run(_load_df(), write_results=False)
    # ⚠️ stamp 喺計算前取：若計算期間 CSV 被重寫（fetch 任務完成），
    # 呢份結果對應嘅係舊文件 → 唔入緩存（下次請求會用新 stamp 重算）。
    if _csv_stamp() != stamp:
        return data
    with _cache_lock:
        _analysis_cache[stamp] = data
    return data


def _get_models(lookback: int) -> list:
    stamp = _csv_stamp()
    # ⚠️ 用 "models" 命名空間做緩存 key：_get_model 會向同一個 (lookback, stamp)
    # 位置寫入 dict，若共用 key，/api/models 會錯誤地回傳 dict（或反之）。
    key = ("models", lookback, stamp)
    with _cache_lock:
        cached = _model_cache.get(key)
        if isinstance(cached, list):
            return cached
    df = _load_df()
    if len(df) <= lookback:
        raise HTTPException(
            status_code=400,
            detail=f"lookback={lookback} 需要多於 {len(df)} 期歷史；數據太少，無法建模。",
        )
    with heavy_lock:
        result = run_all(df, lookback)
    with _cache_lock:
        if len(_model_cache) > 24:
            _model_cache.clear()
        _model_cache[key] = result
    return result


def _get_model(lookback: int) -> dict:
    """單一模型（MLP）結果；同樣有緩存與數據不足保護。"""
    stamp = _csv_stamp()
    key = ("model", lookback, stamp)
    with _cache_lock:
        cached = _model_cache.get(key)
        if isinstance(cached, dict):
            return cached
    df = _load_df()
    if len(df) <= lookback:
        raise HTTPException(
            status_code=400,
            detail=f"lookback={lookback} 需要多於 {len(df)} 期歷史；數據太少，無法建模。",
        )
    with heavy_lock:
        result = model_run(df, lookback, write_result=False)
    with _cache_lock:
        _model_cache[key] = result
    return result


# ───────────────────────── fetch 任務 ─────────────────────────

def _count_rows() -> int:
    """現有主表期數（用嚟計今次新增幾多）。"""
    try:
        with HISTORY_CSV.open(encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _prune_tasks() -> None:
    """清理過期/過多任務（否則 /api/health 的 task 計數同內存會無限增長）。"""
    now = time.time()
    expired = [tid for tid, t in _fetch_tasks.items() if now - t.get("updated_at", 0) > TASK_TTL]
    for tid in expired:
        _fetch_tasks.pop(tid, None)
    if len(_fetch_tasks) > MAX_TASKS:
        for tid in sorted(_fetch_tasks, key=lambda t: _fetch_tasks[t].get("updated_at", 0))[:
                         len(_fetch_tasks) - MAX_TASKS]:
            _fetch_tasks.pop(tid, None)


def _run_task(task_id: str) -> None:
    """後台執行：抓取 HKJC → 整合主表。

    成功/失敗判定以子程序 exit code 為準（唔可以靠「有沒有新增行」）。
    """
    task = _fetch_tasks[task_id]
    out: list[str] = []
    try:
        task.update({"status": "fetching", "progress": "抓取中", "output": []})

        # 1) 由上次數據日期抓到今日
        p1 = subprocess.run(
            [sys.executable, str(SRC / "hkjc_fetch.py"), "--since-last"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE), timeout=FETCH_TIMEOUT,
        )
        out.append(">>> hkjc_fetch.py --since-last")
        out.append(p1.stdout.strip() or p1.stderr.strip() or "(無輸出)")

        if p1.returncode != 0:
            # ⚠️ 抓取失敗 ≠ 該區間無開獎。不能把它當成「更新完成」。
            task.update({
                "status": "error",
                "output": out,
                "error": f"抓取失敗（exit {p1.returncode}）：主表未更新，資料可能已過時",
                "updated_at": time.time(),
            })
            return

        # 2) 整合成主表（build 模式會按日期排序、去重、且不覆寫較完整的舊記錄）
        task.update({"progress": "整合主表"})
        before = _count_rows()
        p2 = subprocess.run(
            [sys.executable, str(SRC / "build_history.py"), "--mode", "build"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE), timeout=BUILD_TIMEOUT,
        )
        out.append(">>> build_history.py --mode build")
        out.append(p2.stdout.strip() or p2.stderr.strip() or "(無輸出)")

        if p2.returncode != 0:
            task.update({
                "status": "error",
                "output": out,
                "error": f"整合失敗（exit {p2.returncode}）",
                "updated_at": time.time(),
            })
            return

        after = _count_rows()
        summary = f"主表 {before} → {after} 期（新增 {after - before} 期）"
        out.append(summary)
        task.update({
            "status": "success",
            "output": out,
            "summary": summary,
            "n_draws": after,
            "added": after - before,
            "updated_at": time.time(),
        })
    except subprocess.TimeoutExpired:
        task.update({
            "status": "error",
            "output": out,
            "error": f"抓取超時（>{FETCH_TIMEOUT}s）：主表未更新",
            "updated_at": time.time(),
        })
    except Exception as exc:  # noqa: BLE001
        task.update({
            "status": "error",
            "output": out,
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at": time.time(),
        })
    finally:
        _fetch_lock.release()


# ───────────────────────── 端點 ─────────────────────────
# 注意：重計算端點用 `def`（不是 `async def`）→ FastAPI 用 threadpool 執行，
# 不會阻塞 event loop；/api/health 之類的輕量端點不會被模型訓練拖慢。

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/analysis")
def api_analysis() -> dict:
    return _get_analysis()


@app.get("/api/model")
def api_model(lookback: int = Query(10, ge=1, le=60)) -> dict:
    return _get_model(lookback)


@app.get("/api/models")
def api_models(lookback: int = Query(10, ge=1, le=60)) -> dict:
    """全部預測模型結果比較（含均勻基線）+ 分析數據（供儀表板渲染）。"""
    return {
        "lookback": lookback,
        "analysis": _get_analysis(),
        "models": _get_models(lookback),
    }


@app.get("/api/run")
@app.post("/api/run")
def api_run(lookback: int = Query(10, ge=1, le=60)) -> dict:
    return {"analysis": _get_analysis(), "model": _get_model(lookback)}


@app.post("/api/fetch")
def api_fetch(x_fetch_token: str | None = Header(default=None, alias="X-Fetch-Token")) -> dict:
    """觸發後台抓取 HKJC + 整合歷史，回傳 task_id 讓前端輪詢。

    ⚠️ 這是一個「抓外部網站 + 重寫主表」的端點。若部署到公開環境：
      - 設 `FETCH_TOKEN` 環境變量，前端必須帶 `X-Fetch-Token` 才能觸發；
      - 同一時間只允許一個任務（單飛），避免併發重寫主表 / 無限抓取。
    """
    required = os.getenv("FETCH_TOKEN", "")
    if not required:
        # ⚠️ 未設 FETCH_TOKEN 時一律停用抓取端點：
        # 這是「抓外部網站 + 重寫 repo 主表」的端點，部署到公開環境若忘記
        # 設 token，任何訪客都能觸發（資源濫用 + 主表被任意重寫）。
        raise HTTPException(
            status_code=503,
            detail="FETCH_TOKEN 未設定：抓取端點已停用（部署時請在環境變數設 FETCH_TOKEN，前端帶 X-Fetch-Token 標頭）。",
        )
    # 常量時間比較，防 timing attack。
    if not hmac.compare_digest(x_fetch_token or "", required):
        raise HTTPException(status_code=403, detail="需要 X-Fetch-Token 標頭")

    if not _fetch_lock.acquire(blocking=False):
        return {"status": "busy", "task_id": None,
                "message": "已有抓取任務進行中，請稍後再試"}

    task_id = uuid.uuid4().hex
    _prune_tasks()
    _fetch_tasks[task_id] = {
        "status": "fetching",
        "progress": "啟動中",
        "output": [],
        "interval": 3,          # 前端輪詢間隔（秒）
        "updated_at": time.time(),
    }
    threading.Thread(target=_run_task, args=(task_id,), daemon=True).start()
    return {"status": "fetching", "task_id": task_id}


@app.get("/api/fetch_status")
def api_fetch_status(task_id: str = Query(..., description="POST /api/fetch 返回嘅 task_id")) -> dict:
    _prune_tasks()
    task = _fetch_tasks.get(task_id)
    if task is None:
        return {"status": "unknown", "error": "沒有此 task_id（可能已過期清理）"}
    return task


@app.get("/api/health")
def health() -> dict:
    """健康檢查 + 數據來源信息（方便確認唔係用 demo 數據）。"""
    _prune_tasks()
    try:
        df = _load_df()
        data_info = {
            "data_source": df.attrs.get("data_source", "unknown"),
            "data_path": df.attrs.get("data_path", ""),
            "n_draws": int(len(df)),
            "dropped_rows": int(df.attrs.get("dropped_rows", 0)),
            "bad_dates": int(df.attrs.get("bad_dates", 0)),
        }
    except HTTPException as exc:
        data_info = {"data_source": "unavailable", "error": exc.detail}
    return {
        "status": "ok",
        "fetch_busy": _fetch_lock.locked(),
        "tasks": len(_fetch_tasks),
        "data": data_info,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
