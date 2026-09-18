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
├── run_all.py                 # 一鍵流程
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
