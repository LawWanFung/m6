# HKJC Mark Six GraphQL API 文件（實測版）

> **本文檔所有內容都經真實 API 實測驗證**（2026-09-21 執行，4390 期完整歷史）。
> 任何改動 query 結構（刪 field、加 field、調換 field 次序、簡化 fragment）
> 都會收到 `Internal server error - WHITELIST_ERROR`。
>
> 舊版 `getJSON.aspx`（HTTP 302 redirect 到主站 SPA）**已失效**，不要再使用。

---

## 1. Endpoint

| 項目 | 值 |
| --- | --- |
| URL | `https://info.cld.hkjc.com/graphql/base/` |
| Method | `POST` |
| Content-Type | `application/json` |

### 必要 Headers

```
Content-Type: application/json
Origin: https://bet.hkjc.com
Referer: https://bet.hkjc.com/
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
Accept-Encoding: gzip
```

回應係 **gzip**；`curl` 用 `--compressed`，Python `requests` 設 `Accept-Encoding: gzip`
（本模組另會檢查 magic bytes `1f 8b` 再 gunzip 一層保險）。

---

## 2. Query（必須一字不改）

完整使用以下 query，**禁止刪減 field、禁止調換 field 順序、禁止自行簡化**：

```graphql
fragment lotteryDrawsFragment on LotteryDraw {
  id
  year
  no
  openDate
  closeDate
  drawDate
  status
  snowballCode
  snowballName_en
  snowballName_ch
  lotteryPool {
    sell
    status
    totalInvestment
    jackpot
    unitBet
    estimatedPrize
    derivedFirstPrizeDiv
    lotteryPrizes {
      type
      winningUnit
      dividend
    }
  }
  drawResult {
    drawnNo
    xDrawnNo
  }
}

query marksixResult($lastNDraw: Int, $startDate: String, $endDate: String, $drawType: LotteryDrawType) {
  lotteryDraws(
    lastNDraw: $lastNDraw
    startDate: $startDate
    endDate: $endDate
    drawType: $drawType
  ) {
    ...lotteryDrawsFragment
  }
}
```

> `src/hkjc_fetch.py` 內嘅 `MARKIX_RESULT_QUERY` 常數就係上面呢段原文。
> GraphQL introspection（`__type` / `__schema`）**被封鎖**，會回
> `"Your query doesn't match the schema."`。

---

## 3. Variables（全部實測）

```json
{
  "lastNDraw": 10,
  "drawType": "All",
  "startDate": null,
  "endDate": null
}
```

| 參數 | 型別 | 實測行為 |
| --- | --- | --- |
| `lastNDraw` | `Int` | 取最近 N 期。**>58 會靜默截斷到 58**（HTTP 200、冇 error）。`0` → 回空。**負數行為不確定，勿用** |
| `drawType` | `LotteryDrawType` | **`"All"` 或 `"SnowBall"`（注意大寫 `B`）**。錯寫 `"Snowball"`／`"snowball"`／`"Foo"` 一律 **HTTP 400** `Your input doesn't match the data type.`。傳 `null` 等同 `"All"` |
| `startDate` | `String` | **`YYYYMMDD`（無 dash）**，例如 `"20260601"`。有 dash（`"2026-06-01"`）會**靜默回空** |
| `endDate` | `String` | 同上 |

### 兩種取數模式

| 模式 | 變數組合 | 特性 |
| --- | --- | --- |
| `lastNDraw` 模式 | `lastNDraw=N`, 日期 `null` | 快；最多 58 期；**實測結果唔連續**（會跳期） |
| 日期範圍模式 | `lastNDraw=null`, 日期 `YYYYMMDD` | **完整、連續**；單次窗口 ≤ 約 3 個月；可由此撈到 1993 年 |

實測變數組合結果：

| `lastNDraw` | `startDate` | `endDate` | 結果 |
| --- | --- | --- | --- |
| 58 | null | null | 58 筆（跨度 2023-09-28 ~ 2026-09-19） |
| 58 | `20260601` | `20260901` | **37 筆**（日期範圍優先，`lastNDraw` 被忽略） |
| null | `20260601` | null | 58 筆（只有一邊日期 → 被忽略，退回 lastNDraw 語意） |
| null | null | `20260901` | 58 筆（同上） |
| null | `""` | `""` | 58 筆（空字串 = 無效） |

---

## 4. 硬限制（實測數據，全部係**靜默失敗**）

| 限制 | 實測 |
| --- | --- |
| `lastNDraw` 上限 | **58 期**。59/60/65 都回 58 筆，**冇任何錯誤** |
| 日期窗口上限 | span **92 日 OK、93 日 → 回空陣列 `[]`**（HTTP 200） |
| 有 dash 日期 | 回空陣列（HTTP 200） |
| `drawType` 錯大小寫 | **HTTP 400**（唯一會明示報錯嘅情況） |

窗口邊界實測（由 2026-06-01 起算）：

| 結束日 | span | 結果 |
| --- | --- | --- |
| 2026-08-31 (+91) | 91 日 | 37 筆 |
| 2026-09-01 (+92) | 92 日 | 37 筆 |
| 2026-09-02 (+93) | 93 日 | **0 筆** |

> ⚠️ 因為超限係「靜默回空」，如果當佢係「該期間無攪珠」就會靜默漏資料。
> `hkjc_fetch._fetch_window()` 因此會**自動二分縮窗重試**，縮到 ≤14 日仍然空
> 才判定為真實缺口。

---

## 5. 完整歷史其實撈得到（推翻舊文檔嘅說法）

用 **日期範圍模式** 逐個 ≤3 個月窗口遍歷，可以拿到由 **1993-01-05** 至今嘅完整歷史：

```
實測：python src/hkjc_fetch.py --from 1993-01-01 --to 2026-09-20
結果：135 個窗口、唯一期數 4390 筆、耗時約 3.5 分鐘（delay 0.4s）
驗證：去重後 4390 筆、格式錯誤 0 筆、年份覆蓋 1993–2026
```

- **最早一期：`1993001N`, 1993-01-05**（1990／1976／1975 都回空 → 數位檔由 1993 年開始）
- 每年期數分佈（真實寫照）：
  - 1993–1999：約 101–114 期／年（**每週 2 期**）
  - 2004–2019：約 149–155 期／年（**每週 3 期**：二／四／六）
  - 2020：只有 30 期 → **COVID 停辦**：2020-02-01 之後停，2020-09-24 復辦
  - 2026：截至 2026-09-19 共 102 期
- `totalInvestment` 早期缺失：1993–1999 全無、2000 只有 35/112、2001 起完整

---

## 6. 回應結構

```
data.lotteryDraws[]            # 陣列，最新一期排最前
├── id                         # e.g. "2026102N"（唯一 key）
├── year                       # "2026"（字串）
├── no                         # 102（期數，Int）
├── openDate / closeDate / drawDate   # "2026-09-19+08:00"
├── status                     # "Result" = 已開獎（其他狀態未開獎，應跳過）
├── snowballCode               # 金多寶代碼，非金多寶為 ""（全歷史 151 期有值）
├── snowballName_en / _ch      # e.g. "Summer Snowball" / "暑期金多寶"
├── lotteryPool
│   ├── sell                   # Bool
│   ├── status                 # "Payout" …
│   ├── totalInvestment        # 總投注額（字串數字）
│   ├── jackpot / unitBet / estimatedPrize / derivedFirstPrizeDiv
│   └── lotteryPrizes[]        # type 1..7 = 頭獎…七獎
│       ├── type / winningUnit / dividend
└── drawResult
    ├── drawnNo                # 6 個正碼（array of int）
    └── xDrawnNo               # 特別號碼（int）
```

### 獎項 type 對照

| type | 獎項 | 主表欄位 |
| --- | --- | --- |
| 1 | 頭獎 | `p1` / `p1_units` |
| 2 | 二獎 | `p2` / `p2_units` |
| 3 | 三獎 | `p3` / `p3_units` |
| 4 | 四獎 | `p4` / `p4_units` |
| 5 | 五獎 | `p5` / `p5_units` |
| 6 | 六獎 | `p6` / `p6_units` |
| 7 | 七獎 | `p7` / `p7_units` |

### GraphQL → 主表欄位映射（`src/hkjc_fetch.py::normalize`）

| 主表欄位 | 來源 |
| --- | --- |
| `draw_id` | `id` |
| `date` | `drawDate` → `DD/MM/YYYY` |
| `numbers` | `drawResult.drawnNo`（排序後 6 個） |
| `extra_ball` | `drawResult.xDrawnNo` |
| `snowball_code` | `snowballCode` |
| `snowball_name_ch` / `_en` | `snowballName_ch` / `snowballName_en` |
| `total_turnover` | `lotteryPool.totalInvestment`（轉 float） |
| `p1..p7` | `lotteryPrizes[type=N].dividend` |
| `p1_units..p7_units` | `lotteryPrizes[type=N].winningUnit` |

---

## 7. 用法

```bash
# 最近 50 期（快，lastNDraw 模式）
python src/hkjc_fetch.py --last-n 50

# 增量更新（最近 60 日，日期範圍模式 → 完整連續）
python src/hkjc_fetch.py --latest

# 完整歷史（分段寫入 data/raw/seg_*.json，可中斷續傳）
python src/hkjc_fetch.py --from 1993-01-01
python src/hkjc_fetch.py --from 2020-01-01 --to 2020-12-31   # 指定範圍
python src/hkjc_fetch.py --from 1993-01-01 --force           # 忽略舊分段重抓

# 只要金多寶期
python src/hkjc_fetch.py --last-n 30 --draw-type SnowBall

# 整合成主表（去重）
python src/build_history.py --mode build
```

### 錯誤訊息對照

| 訊息 | 原因 | 處理 |
| --- | --- | --- |
| `Internal server error - WHITELIST_ERROR` | query 結構被改動 | 用回第 2 節原文 |
| `Your input doesn't match the data type.`（HTTP 400） | `drawType` 大小寫錯 | 用 `"All"` / `"SnowBall"` |
| HTTP 200 但 `lotteryDraws: []` | 日期格式有 dash／窗口 >92 日／該期無攪珠 | 用 `YYYYMMDD` 及 ≤3 個月窗口；本模組會自動二分縮窗 |
| 期數少於預期 | `lastNDraw` > 58（靜默截斷） | 要完整資料請用日期範圍模式 |

### curl 測試

```bash
# repo 根目錄嘅 q.json 就係實際 payload（已驗證）
curl -sL --compressed -X POST 'https://info.cld.hkjc.com/graphql/base/' \
  -H 'Content-Type: application/json' \
  -H 'Origin: https://bet.hkjc.com' \
  -H 'Referer: https://bet.hkjc.com/' \
  -H 'Accept-Encoding: gzip' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' \
  --data-binary @q.json
```

---

## 8. 相關 Endpoints（HKJC 其他服務）

| 服務 | URL |
| --- | --- |
| GraphQL | `https://info.cld.hkjc.com/graphql/base/` |
| Info API | `https://infoapi.hkjc.com` |
| Notes | `https://notes.hkjc.com` |
| Special | `https://special.hkjc.com/e-win` |

---

## 9. 修訂記錄

- **2025-11-24** 首次發現 GraphQL endpoint（舊文檔：`drawType: "Snowball"`、`startDate/endDate` 無效、無法撈完整歷史）。
- **2026-09-21 實測修正**（本次）：
  1. `drawType` 正確值係 **`"SnowBall"`**（大寫 B），`"Snowball"` → HTTP 400。
  2. `startDate` / `endDate` **有效**，格式 **`YYYYMMDD`**、需配 `lastNDraw: null`。
  3. 日期窗口上限 **≈3 個月**（span 92 日 OK／93 日回空），超限**靜默回空**。
  4. **完整歷史可撈**：1993-01-05 起，實測 4390 期。
  5. `lastNDraw` 超 58 係**靜默截斷**（非報錯），且結果**唔連續**。
  6. 記錄 2020 年 COVID 真實空窗（2020-02-01 → 2020-09-24）。
