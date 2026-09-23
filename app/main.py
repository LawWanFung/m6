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
import datetime as dt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 讓 app 能找到 src / analyze / model 模組
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

import analyze.frequency as freqmod
from model.predict import run as model_run

TEMPLATES_DIR = Path(__file__).parent / "templates"
SRC = BASE / "src"
#: 真實歷史主表（分析／模型唯一數據來源；已入 repo）
HISTORY_CSV = BASE / "data" / "mark6_history.csv"
SAMPLE_CSV = BASE / "data" / "mark6_sample.csv"

# 後台 fetch 任務登記（task_id -> 狀態）
_fetch_tasks: dict[str, dict] = {}

app = FastAPI(title="Mark Six 儀表板", version="1.0.0")


def _before_count() -> int:
    """現有主表期數（用嚟計今次新增幾多）。"""
    try:
        return len(pd.read_csv(HISTORY_CSV))
    except Exception:  # noqa: BLE001
        return 0


def _run_task(task_id: str) -> None:
    """背景執行：由「上次數據日期」抓到今日，然後整合成主表。

    步驟：
      1. `hkjc_fetch.py --since-last`：自動判斷主表最後一期日期 → 抓到今日
      2. `build_history.py --mode build`：把所有 raw JSON 去重整合成主表
    """
    fetch_script = SRC / "hkjc_fetch.py"
    build_script = SRC / "build_history.py"

    out: list[str] = []
    try:
        before = _before_count()
        _fetch_tasks[task_id].update(
            {"status": "fetching", "progress": "抓取中", "output": []}
        )

        # 1) 由上次數據日期抓到今日
        # 子程序強制 UTF-8 輸出，這裡亦要明確用 UTF-8 解碼（否則 Windows cp950 locale 會炸）
        p1 = subprocess.run(
            [sys.executable, str(fetch_script), "--since-last"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE),
        )
        out.append(">>> hkjc_fetch.py --since-last")
        out.append(p1.stdout.strip() or p1.stderr.strip() or "(無輸出)")

        # 2) 整合成主表（去重排序；--mode build 只讀本地 raw，唔會用 demo 數據）
        _fetch_tasks[task_id].update({"progress": "整合主表"})
        p2 = subprocess.run(
            [sys.executable, str(build_script), "--mode", "build"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE),
        )
        out.append(">>> build_history.py --mode build")
        out.append(p2.stdout.strip() or p2.stderr.strip() or "(無輸出)")

        after = _before_count()
        summary = f"主表 {before} → {after} 期（新增 {after - before} 期）"
        out.append(summary)

        if p1.returncode == 0 and p2.returncode == 0:
            _fetch_tasks[task_id].update(
                {
                    "status": "success",
                    "output": out,
                    "summary": summary,
                    "n_draws": after,
                    "updated_at": time.time(),
                }
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
    """載入**真實**歷史數據（data/mark6_history.csv）。

    明確唔傳 `allow_sample`：主表不存在就報錯，**絕對不會**退回合成樣本，
    以免儀表板顯示 demo 數據而令人誤以為係真實開獎結果。
    """
    import analyze.frequency as freqmod

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
        raise HTTPException(status_code=503, detail="數據來源不是真實 HKJC 數據，已中止分析。")
    return df


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


def _load_models(lookback: int = 10) -> list:
    """執行全部預測模型，返回有序結果列表（含均勻基線）。"""
    from model.registry import run_all

    return run_all(_load_df(), lookback)


@app.get("/api/models")
async def api_models(lookback: int = Query(10, ge=1, le=60)) -> dict:
    """所有預測模型的結果比較（含均勻基線）＋分析數據（供儀表板渲染）。"""
    return {
        "lookback": lookback,
        "analysis": _load_analysis(),
        "models": _load_models(lookback),
    }


@app.get("/api/run")
@app.post("/api/run")
async def api_run(lookback: int = Query(10, ge=1, le=60)) -> dict:
    return {"analysis": _load_analysis(), "model": _load_model(lookback)}


@app.post("/api/fetch")
async def api_fetch() -> dict:
    """觸發背景抓取 HKJC + 整合歷史，回傳 task_id 讓前端輪詢。"""
    task_id = uuid.uuid4().hex
    # 一開始就標記 'fetching'，否則前端首次輪詢會見到 'starting' 而誤判完成
    _fetch_tasks[task_id] = {
        "status": "fetching", "progress": "啟動中", "output": [], "updated_at": time.time(),
    }
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
    """健康檢查 + 數據來源資訊（方便確認唔係用 demo 數據）。"""
    import csv

    try:
        import csv

        with HISTORY_CSV.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # date 係 DD/MM/YYYY，唔可以字串排序
        parsed = []
        for r in rows:
            try:
                d, m, y = str(r.get("date", "")).split("/")
                parsed.append(dt.date(int(y), int(m), int(d)))
            except (ValueError, TypeError):
                continue
        data_info = {
            "data_source": "real",
            "data_path": "data/mark6_history.csv",
            "n_draws": len(rows),
            "first": min(parsed).isoformat() if parsed else None,
            "last": max(parsed).isoformat() if parsed else None,
        }
    except Exception as exc:  # noqa: BLE001
        data_info = {"data_source": "unavailable", "error": str(exc)}
    return {"status": "ok", "tasks": len(_fetch_tasks), "data": data_info}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
