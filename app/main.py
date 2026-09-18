"""
六合彩數據儀表板 — FastAPI Web App
===================================ploy 會將流量轉發到此端口。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 讓 app 能找到 src / analyze / model 模組
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pandas as pd

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse

import analyze.frequency as freqmod
from model.predict import run as model_run

TEMPLATES_DIR = Path(__file__).parent / "templates"
SRC = BASE / "src"

# 後台 fetch 任務登記（task_id -> 狀態）
_fetch_tasks: dict[str, dict] = {}

app = FastAPI(title="Mark Six 儀表板", version="1.0.0")


def _run_task(task_id: str) -> None:
    """在背景線程執行 hkjc_fetch + build_history，把結果寫入 _fetch_tasks。"""
    fetch_script = SRC / "hkjc_fetch.py"
    build_script = SRC / "build_history.py"

    out: list[str] = []
    try:
        _fetch_tasks[task_id].update({"status": "fetching", "output": []})

        # 1) 抓取最新一期 HKJC 結果
        p1 = subprocess.run(
            [sys.executable, str(fetch_script), "--latest"],
            capture_output=True, text=True, cwd=str(BASE),
        )
        out.append(">>> hkjc_fetch.py --latest")
        out.append(p1.stdout.strip() or p1.stderr.strip() or "(無輸出)")

        # 2) 整合歷史
        p2 = subprocess.run(
            [sys.executable, str(build_script), "--mode", "update"],
            capture_output=True, text=True, cwd=str(BASE),
        )
        out.append(">>> build_history.py --mode update")
        out.append(p2.stdout.strip() or p2.stderr.strip() or "(無輸出)")

        if p1.returncode == 0 and p2.returncode == 0:
            _fetch_tasks[task_id].update(
                {"status": "done", "output": out, "updated_at": time.time()}
            )
        else:
            _fetch_tasks[task_id].update(
                {
                    "status": "error",
                    "output": out,
                    "error": f"exit codes {p1.returncode}, {p2.returncode}",
                    "updated_at": time.time(),
                }
            )
    except Exception as exc:  # noqa: BLE001
        _fetch_tasks[task_id].update(
            {"status": "error", "output": out, "error": str(exc), "updated_at": time.time()}
        )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


def _load_df() -> pd.DataFrame:
    import analyze.frequency as freqmod

    return freqmod.load()


def _load_analysis() -> dict:
    import analyze.frequency as freqmod

    return freqmod.run(_load_df())


def _load_model(lookback: int = 10) -> dict:
    from model.predict import run as model_run

    return model_run(_load_df(), lookback)


@app.get("/api/analysis")
async def api_analysis() -> dict:
    return _load_analysis()


@app.get("/api/model")
async def api_model() -> dict:
    return _load_model(10)


@app.get("/api/run")
@app.post("/api/run")
async def api_run(lookback: int = Query(10, ge=1, le=60)) -> dict:
    return {"analysis": _load_analysis(), "model": _load_model(lookback)}


@app.post("/api/fetch")
async def api_fetch() -> dict:
    """觸發背景抓取 HKJC + 整合歷史，回傳 task_id 讓前端輪詢。"""
    task_id = uuid.uuid4().hex
    _fetch_tasks[task_id] = {"status": "starting", "output": [], "updated_at": time.time()}
    threading.Thread(target=_run_task, args=(task_id,), daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/fetch_status")
async def api_fetch_status(task_id: str) -> dict:
    task = _fetch_tasks.get(task_id)
    if task is None:
        return {"status": "unknown", "error": "沒有此 task_id"}
    task["updated_at"] = time.time()  # 保持存活
    return task


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "tasks": len(_fetch_tasks)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
