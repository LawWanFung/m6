# 六合彩（Mark Six）歷史數據 + 統計分析 + 建模

> ⚠️ **重要前提**：六合彩係**獨立隨機事件**，任何基於歷史號碼的模型都**無法真正預測**下一期號碼。
> 本專案用於**統計分析、流程練習與「無法預測」之統計驗證**，不構成任何預測或投注建議。
> 博彩有風險，請合法理性參與。

## 專案結構

```
mark6/
├── data/
│   ├── mark6_history.csv      # ⭐ 真實歷史主表（1993-01-05 起 4390 期，已入 repo）
│   ├── mark6_sample.csv       # 合成樣本（500 期，只供離線測試，App 唔會用）
│   └── raw/                   # HKJC 原始 JSON 快取（不入 repo）：seg_*.json 分段、last_all.json 最新
├── src/
│   ├── hkjc_fetch.py          # 抓取 HKJC 官方 GraphQL API（見 docs/hkjc_graphql_schema.md）
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

### 2. 抓取真實數據（需網絡可直連 HKJC）
HKJC 官方 API：`POST https://info.cld.hkjc.com/graphql/base/`（詳見 `docs/hkjc_graphql_schema.md`）

> ⚠️ 兩種模式（實測，詳見 `docs/hkjc_graphql_schema.md`）：
> * `lastNDraw`：最多 **58 期**（超過**靜默截斷**），且結果**唔連續**。
> * 日期範圍（`YYYYMMDD`，窗口 **≤ 3 個月**）：**完整、連續**，可撈到 1993 年。
>   超限係**靜默回空**，所以模組會自動二分縮窗重試。

> 📦 **完整歷史（1993-01-05 起、4390 期）已經入 repo**：`data/mark6_history.csv`。
> 所以 clone 完就即刻有真實數據，唔需要出網就跑到分析／儀表板。
> `data/raw/`（GraphQL 原始回應快取）唔入 repo，可以隨時重新抓。

```bash
# 完整歷史（約 3.5 分鐘；分段寫入 data/raw/seg_*.json，可中斷續傳）
python src/hkjc_fetch.py --from 1993-01-01

# 抓最近 50 期（快，lastNDraw 模式）→ data/raw/last_all.json
python src/hkjc_fetch.py --last-n 50

# 整合成主表（讀光所有 raw JSON，按 draw_id 去重排序；可重覆執行）
python src/build_history.py --mode build
```

### 3. 增量更新（每期開獎之後跑一次）

`--since-last` 會**自動判斷「對上一次數據」係邊一日**（先讀 `data/mark6_history.csv`
嘅最大日期，冇有就掃 `data/raw/*.json`），然後由該日一直抓到**今日**：

```bash
python src/hkjc_fetch.py --since-last   # 上次數據 → 今日（寫入 last_all.json）
python src/build_history.py --mode build # 去重後追加新期數
python analyze/frequency.py
```

- `--latest` 係 `--since-last` 嘅同義詞（Web App 按鈕就係跑這個）。
- 完全冇舊數據（主表＋raw 都空）→ 自動改為由 1993 做完整抓取。
- 想固定回溯日數可用 `--days 90`（會覆寫自動判斷）。

## 冷熱號與公平性（歷史上場窗口）

**冷熱號按『率』排名**（每期出現機率），**已考慮號碼歷史上場窗口**：
46–47 號 1996-06-11、48–49 號 2002-07-04 先加入。若用全歷史 raw count
會因為「出現得少」而被誤判為冷號。卡方公平性檢驗亦按三代（pool 45/47/49）
分組做，避免遲出號碼產生假性偏差。實作見 `analyze/frequency.py`。

## Web App（FastAPI + Dokploy）

`app/` 係一個 FastAPI 網頁應用，把統計分析與建模結果做成儀表板：

| 端點 | 說明 |
|------|------|
| `GET /` | 儀表板（頻率圖、卡方檢驗、冷熱號、多模型比較） |
| `GET /api/analysis` | 統計分析 JSON |
| `GET /api/model` | 單模型（MLP）結果 JSON（向後相容） |
| `GET /api/models?lookback=N` | **所有預測模型的結果比較**（含均勻基線）+ 分析數據 |
| `POST /api/run?lookback=N` | 重新執行分析 + 單模型建模 |
| `POST /api/fetch` | 背景抓取：由「上次數據日期」抓到今日，再整合成主表（回 `task_id` 輪詢） |
| `GET /api/fetch_status?task_id=…` | 抓取進度（`fetching` / `success` / `error`） |
| `GET /api/health` | 健康檢查 + **數據來源資訊**（期數、日期範圍） |

### ⚠️ 數據來源保證（唔准用 demo 數據）

分析／建模／儀表板**只會用真實 HKJC 數據** `data/mark6_history.csv`：

- `analyze.frequency.load()` 預設 `allow_sample=False`：主表不存在或為空就直接報錯，
  **唔會靜默退回合成樣本** `mark6_sample.csv`。
- 只有明確傳 `allow_sample=True`（`run_all.py` 離線測試、CI）才會用合成樣本，
  並且 `df.attrs["data_source"]` 會標記為 `"sample"`。
- API 回應會帶 `data_source` / `date_range`，儀表板會顯示
  `✅ 真實 HKJC 數據（1993-01-05 ~ 2026-09-19）`；若來源不是 `real`，
  `/api/analysis` 直接回 **HTTP 503**，唔會顯示假數據。

### lookback 係咩？

`lookback`（滯後期數，預設 10）＝**餵給模型嘅歷史窗口大細**。
建模時，每一期嘅特徵就係「對上 `lookback` 期、每個號碼（1–49）出現過幾多次」
→ 一條 `lookback × 49` 維嘅向量，用 MLP 去預測當期 6 個號碼。

| lookback | 效果 |
| --- | --- |
| 細（如 3） | 特徵少、訓練快，但只看近期，訊息量少 |
| 大（如 60） | 特徵多（2940 維）、訓練慢，噪聲多、容易過擬合 |

重點係：**無論點調 lookback，模型 log-loss（~8.07）都輸給均勻 baseline（ln 49 ≈ 3.89）**。
即係話歷史頻率對預測下一期**零作用**——呢個正正係「六合彩係獨立隨機事件」嘅實證。
所以 `lookback` 嘅用途係**示範同驗證「模型無用」**，唔係用嚟調到「有得贏」。

### 本地啟動
```bash
pip install -r requirements.txt
PORT=8000 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打開 http://localhost:8000
```
> ⚠️ 本環境（pi）本身佔用 host 的 `8000` 端口，本地測試請換 port，
> 例如 `PORT=8137 python -m uvicorn app.main:app --port 8137`。
> Docker 內部獨立，`PORT=8000` 無問題。

### 部署到 Dokploy（docker-compose 方式）
Dokploy 用 `docker-compose.yml` 部署，檔案如下：

```yaml
services:
  mark6:
    build: .
    environment:
      - PORT=${PORT:-8000}
```

部署步驟：
1. Dokploy 新增 Deployment → 類型選 **docker-compose**
2. 指定本檔案所在目錄（通常即 repo root）
3. 為 `mark6` 這個 service 指定域名
4. Dokploy 會把該域名路由到 `mark6:${PORT}`（預設 8000）

> `docker-compose.yml` 同 `.env`（`PORT=8000`）一齊放 repo root，
> docker-compose 會自動用 `.env` 做變數替換。

### 為什麼 `/` 直接返回 HTML 而非用 Jinja 伺服器端渲染？
此環境的 Starlette 1.2.x 與 Jinja2 3.1.x 在模板快取上有版本相容問題，
故前端圖表全部由 JS 向 `/api/*` 拉取資料，`/` 僅返回 HTML 檔案，
避開此相容性陷阱。

## 各模組說明

| 檔案 | 用途 |
|------|------|
| `src/hkjc_fetch.py` | 官方 GraphQL API 抓取：完整歷史（日期範圍分段）或增量 `--since-last`；`normalize()` 轉主表欄位 |
| `src/build_history.py` | 合併所有 JSON → 主表 CSV（按 `draw_id` 去重），支援增量追加 |
| `analyze/frequency.py` | 號碼頻率、卡方檢驗（是否均勻/公平）、冷熱號、奇偶/大小比 |
| `model/base.py` | 共用特徵工程（`parse_numbers`、`multiclass_samples`）與模型基底介面 |
| `model/ml_models.py` | sklearn 系列：邏輯回歸、隨機森林、高斯貝葉斯、MLP（共用評估骨架） |
| `model/stat_models.py` | 統計／序列模型：頻率法（熱號）、馬爾可夫鏈趨勢 |
| `model/registry.py` | **模型Registry**：統一執行全部模型，供 `/api/models` 與儀表板列出 |
| `model/predict.py` | 單模型（MLP）基底，用 log-loss 對比均勻隨機 baseline（向後相容） |

## 為什麼模型「預測」會輸給隨機？

`model/predict.py` 用過去 10 期號碼頻率作特徵訓練 MLP，預測下期各號碼機率，
再以 log-loss 對比「均勻分佈 baseline（= ln(49) ≈ 3.89）」。
結果模型 log-loss **高於** baseline —— 這正是六合彩不可預測的統計證據。

## 多預測模型比較（儀表板）

`GET /api/models?lookback=N` 會**同時執行全部模型**，由儀表板統一列出比較表
（`🧠 多預測模型比較`）。所有模型都用同一個 metric（log-loss，越低越好），
並以均勻基線（`ln 49 ≈ 3.89`）為參考線：

| 模型 | 類型 | 思路 | 為何也贏不了隨機 |
|------|------|------|----------------|
| 均勻基線 Uniform | baseline | 理論參考線（log-loss = ln 49） | — |
| 邏輯回歸 Logistic | sklearn | 線性權重組合 | 歷史頻率與下期無線性關係 |
| 神經網絡 MLP | 深度學習 | 多層感知機非線性映射 | 無信號可學，過度擬合噪聲 |
| 隨機森林 Random Forest | sklearn | 多棵决策樹集成 | 樹只能記憶歷史，泛化至零 |
| 高斯樸素貝葉斯 | sklearn | 常態假設 + 貝葉斯後驗 | 獨立假設與實際不符，且無信號 |
| 頻率法 熱號 Frequency | 統計 | 長週期熱號（賭徒謬誤） | 每期均等，熱號無預測力 |
| 馬爾可夫鏈 趨勢 Markov | 統計 | 每號碼存在／不存在轉移 | 轉移機率趨向常態，無預測力 |

所有模型的 log-loss 都 **≥ 均勻基線**，再次印證「六合彩無法預測」。

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
