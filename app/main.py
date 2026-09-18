"""
六合彩數據儀表板 — FastAPI Web App
===================================

部署到 Dokploy（Docker）後：
  GET /              儀表板（頻率、卡方檢驗、冷熱號、建模結果）
  GET /api/analysis  統計分析 JSON
  GET /api/model     建模結果 JSON
  POST /api/run      重新執行分析 + 建模（可傳 ?lookback=10）

監聽 PORT（預設 8000），Dokploy 會將流量轉發到此端口。
"""

from __future__ import annotations

import os
import sys
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

app = FastAPI(title="Mark Six 儀表板", version="1.0.0")

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_df() -> pd.DataFrame:
    return freqmod.load()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    # 儀表板前端由 JS 直接向 /api/* 拉取資料，故此處只需返回 HTML 檔案，
    # 避開 Starlette/Jinja2 伺服器端模板渲染的版本相容問題。
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/analysis")
async def api_analysis() -> dict:
    df = _load_df()
    out = freqmod.run(df)
    return out


@app.get("/api/model")
async def api_model() -> dict:
    df = _load_df()
    lookback = 10
    return model_run(df, lookback)


@app.get("/api/run")
@app.post("/api/run")
async def api_run(lookback: int = Query(10, ge=1, le=60)) -> dict:
    df = _load_df()
    analysis = freqmod.run(df)
    model = model_run(df, lookback)
    return {"analysis": analysis, "model": model}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
