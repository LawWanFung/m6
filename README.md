# 六合彩（Mark Six）歷史數據 + 統計分析 + 建模

> ⚠️ **重要前提**：六合彩係**獨立隨機事件**，任何基於歷史號碼的模型都**無法真正預測**下一期號碼。
> 本專案用於**統計分析、流程練習與「無法預測」之統計驗證**，不構成任何預測或投注建議。
> 博彩有風險，請合法理性參與。

## 專案結構

```
mark6/
├── data/
│   ├── mark6_history.csv      # 主表（真實數據整合後放這裡）
│   ├── mark6_sample.csv       # 合成樣本（500 期，本地測試用）
│   └── raw/                   # HKJC 原始 JSON（每 3 個月一檔）
├── src/
│   ├── hkjc_fetch.py          # 抓取 HKJC 官方 JSON API（按 3 個月分段）
│   ├── build_history.py       # 整合原始 JSON → 主表 CSV（去重、增量）
│   └── gen_sample.py          # 生成合成樣本數據
├── analyze/
│   └── frequency.py           # 頻率分析 + 卡方公平性檢驗 + 描述性統計
├── model/
│   └── predict.py             # 特徵工程 + MLP 分類器 + 與均勻 baseline 比較
├── app/                       # ⭐ FastAPI Web App（供 Dokploy 部署）
│   ├── main.py                #   路由：/ 儀表板 + /api/* 資料
│   └── templates/index.html   #   Chart.js 儀表板前端
├── run_all.py                 # 一鍵流程
├── Dockerfile                 # ⭐ Dokploy Docker 部署
├── .dockerignore
└── requirements.txt
```

## 快速開始

### 1. 離線測試（合成數據）
```bash
pip install -r requirements.txt
python run_all.py
```

### 2. 抓取真實歷史數據（需網絡可直連 HKJC）
HKJC 官方 API：`http://bet.hkjc.com/marksix/getJSON.aspx?sd=YYYYMMDD&ed=YYYYMMDD&sb=0`
```bash
# 抓取 1993 年至今全部歷史
python src/hkjc_fetch.py --from 1993-01-01 --to 2025-12-31
python src/build_history.py --mode build      # 整合成主表
```

### 3. 增量更新（每期之後跑一次）
```bash
python src/hkjc_fetch.py --latest
python src/build_history.py --mode update
python analyze/frequency.py
```

## Web App（FastAPI + Dokploy）

`app/` 係一個 FastAPI 網頁應用，把統計分析與建模結果做成儀表板：

| 端點 | 說明 |
|------|------|
| `GET /` | 儀表板（頻率圖、卡方檢驗、冷熱號、建模結果） |
| `GET /api/analysis` | 統計分析 JSON |
| `GET /api/model` | 建模結果 JSON |
| `POST /api/run?lookback=N` | 重新執行分析 + 建模 |

### 本地啟動
```bash
pip install -r requirements.txt
PORT=8000 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打開 http://localhost:8000
```
> ⚠️ 本環境（pi）本身佔用 host 的 `8000` 端口，本地測試請換 port，
> 例如 `PORT=8137 python -m uvicorn app.main:app --port 8137`。
> Docker 內部獨立，`PORT=8000` 無問題。

### 部署到 Dokploy（Docker 方式）
1. Dokploy 新增 Web Service，選此 repo。
2. Dockerfile 已設定：依賴自動安裝，監聽 `$PORT`（預設 8000）。
3. Dokploy 會傳入 `PORT` 環境變數，`CMD` 會跟住它。
4. 部署後打開分配的域名即可。

### 為什麼 `/` 直接返回 HTML 而非用 Jinja 伺服器端渲染？
此環境的 Starlette 1.2.x 與 Jinja2 3.1.x 在模板快取上有版本相容問題，
故前端圖表全部由 JS 向 `/api/*` 拉取資料，`/` 僅返回 HTML 檔案，
避開此相容性陷阱。

## 各模組說明

| 檔案 | 用途 |
|------|------|
| `src/hkjc_fetch.py` | 官方 API 抓取，每 3 個月分段遍歷，逐區間存 JSON，易斷點續傳 |
| `src/build_history.py` | 合併所有 JSON → 主表 CSV（按 `draw_id` 去重），支援增量追加 |
| `analyze/frequency.py` | 號碼頻率、卡方檢驗（是否均勻/公平）、冷熱號、奇偶/大小比 |
| `model/predict.py` | 滯後頻率特徵 + MLP，用 log-loss 對比均勻隨機 baseline |

## 為什麼模型「預測」會輸給隨機？

`model/predict.py` 用過去 10 期號碼頻率作特徵訓練 MLP，預測下期各號碼機率，
再以 log-loss 對比「均勻分佈 baseline（= ln(49) ≈ 3.89）」。
結果模型 log-loss **高於** baseline —— 這正是六合彩不可預測的統計證據。

## 輸出結果

- `analyze/results/analysis.json` — 頻率、卡方、冷熱號、描述性統計
- `model/results/model_result.json` — 模型 log-loss 與可預測性判定

## Web App 檔案

| 檔案 | 用途 |
|------|------|
| `app/main.py` | FastAPI 路由（`/` + `/api/*`），讀 `PORT` env |
| `app/templates/index.html` | Chart.js 儀表板，前端 JS 拉 `/api/*` |
| `Dockerfile` | 依 `$PORT` 啟動 uvicorn，供 Dokploy 部署 |
| `.dockerignore` | 部署時排除不必要檔案 |
