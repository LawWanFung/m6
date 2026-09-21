"""
HKJC Mark Six 數據抓取模組（GraphQL）
=====================================

數據來源：香港賽馬會官方 GraphQL API
    POST https://info.cld.hkjc.com/graphql/base/

舊版 JSON API（`http://bet.hkjc.com/marksix/getJSON.aspx`，已 302 redirect 到主站 SPA）
**已失效**。完整規格（含實測證據）見 docs/hkjc_graphql_schema.md。

⚠️ `MARKIX_RESULT_QUERY` 一字不能改，否則會收到
   `Internal server error - WHITELIST_ERROR`。

兩種取數模式（實測）：
  1. `lastNDraw` 模式：取最近 N 期，**靜默**截斷於 58 期（多過都只回 58，冇 error）。
  2. 日期範圍模式：`startDate` / `endDate` 用 **`YYYYMMDD`（無 dash）**，
     窗口最多約 3 個月（實測 92 日 OK、93 日回空），
     可由此由 1993-01-05 一路撈到最新 → **完整歷史係拿到嘅**。
     ⚠️ 超過窗口限制係**靜默回空陣列**（HTTP 200 + `[]`），唔會報錯，
        所以本模組會自動偵測並用二分法縮窗重試。

用法：
    python src/hkjc_fetch.py                          # 抓最近 50 期 -> data/raw/last_all.json
    python src/hkjc_fetch.py --last-n 30              # 抓最近 30 期
    python src/hkjc_fetch.py --latest                 # 增量更新（最近 60 日）
    python src/hkjc_fetch.py --from 1993-01-01        # 完整歷史（分段，可中斷續傳）
    python src/hkjc_fetch.py --draw-type SnowBall     # 只取金多寶期

注意：
  * 請遵守 HKJC 服務條款，控制請求頻率（預設每次請求暫停 0.8 秒）。
  * 若網絡環境封鎖 HKJC，需在能直連該網站的機器上執行。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import requests

# 統一以 UTF-8 輸出（唔做嘅話 subprocess capture_output 會用 locale 編碼解碼失敗）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

# ---------------------------------------------------------------- 常數 / 設定

HOST = "https://info.cld.hkjc.com"
GRAPHQL_URL = f"{HOST}/graphql/base/"

#: 專案根目錄（src/ 嘅上一層）
BASE_DIR = Path(__file__).resolve().parent.parent
#: 歷史主表（build_history.py 產出；真實歷史已入 repo）
HISTORY_CSV = BASE_DIR / "data" / "mark6_history.csv"
#: 原始 JSON 快取目錄（預設）
RAW_DIR = BASE_DIR / "data" / "raw"

#: `lastNDraw` 實測硬上限（>58 會「靜默」截斷到 58，冇錯誤訊息）
MAX_LAST_N_DRAW = 58
#: 建議值
RECOMMENDED_LAST_N_DRAW = 50
DEFAULT_LAST_N_DRAW = RECOMMENDED_LAST_N_DRAW

#: 日期範圍模式嘅 drawType 值（注意大寫 B；錯寫 "Snowball" 會 HTTP 400）
DRAW_TYPE_ALL = "All"
DRAW_TYPE_SNOWBALL = "SnowBall"
DRAW_TYPES = (DRAW_TYPE_ALL, DRAW_TYPE_SNOWBALL)
#: 兼容舊寫法（會自動糾正並提示）
_DRAW_TYPE_ALIASES = {"snowball": DRAW_TYPE_SNOWBALL, "snowball ": DRAW_TYPE_SNOWBALL}

#: 日期範圍窗口上限（實測：span 92 日 OK、93 日靜默回空）。
#: 預設用「對齊月頭嘅 3 個日曆月」，span 最多 92 日。
WINDOW_MONTHS = 3
#: 窗口絕對上限（span 日數）；超過就自動切兩半（**絕不截斷，避免靜默漏資料**）
MAX_WINDOW_SPAN_DAYS = 92
#: 二分法縮窗下限（日）：細過此值仍然係空 → 判定為真實資料缺口
MIN_WINDOW_DAYS = 14

#: 歷史資料最早一期（實測 1993-01-05；1990/1976/1975 都回空）
EARLIEST_DATE = dt.date(1993, 1, 1)

DEFAULT_DELAY = 0.8  # 秒，避免請求過頻
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# ⚠️ WHITELIST：以下 query 一字不能改（fragment 必須完整、field 次序不可動）。
MARKIX_RESULT_QUERY = """fragment lotteryDrawsFragment on LotteryDraw {
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
"""

HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip",
}


# ---------------------------------------------------------------- 工具


def normalize_draw_type(draw_type: str | None) -> str:
    """糾正 drawType 大小寫（`Snowball` -> `SnowBall`）。"""
    if not draw_type:
        return DRAW_TYPE_ALL
    if draw_type in DRAW_TYPES:
        return draw_type
    fixed = _DRAW_TYPE_ALIASES.get(str(draw_type).strip().lower())
    if fixed:
        print(f"  [!] drawType={draw_type!r} 大小寫有誤，已自動改用 {fixed!r}。", file=sys.stderr)
        return fixed
    print(f"  [!] 未知 drawType {draw_type!r}，改用 {DRAW_TYPE_ALL!r}。", file=sys.stderr)
    return DRAW_TYPE_ALL


def clamp_last_n(last_n: int | None) -> int:
    """把 lastNDraw 夾到 1..MAX_LAST_N_DRAW。"""
    if last_n is None:
        return DEFAULT_LAST_N_DRAW
    try:
        n = int(last_n)
    except (TypeError, ValueError):
        return DEFAULT_LAST_N_DRAW
    if n < 1:
        return 1
    if n > MAX_LAST_N_DRAW:
        print(
            f"  [!] lastNDraw={n} 超過實測硬上限 {MAX_LAST_N_DRAW}（API 會靜默截斷），"
            f"已夾到 {RECOMMENDED_LAST_N_DRAW}（建議值）。",
            file=sys.stderr,
        )
        return RECOMMENDED_LAST_N_DRAW
    return n


def _ymd(d: dt.date) -> str:
    """API 要 `YYYYMMDD`（無 dash）；有 dash 會靜默回空。"""
    return d.strftime("%Y%m%d")


def _add_months(d: dt.date, months: int) -> dt.date:
    """月份加減（日期設為 1 號，避免月底溢出）。"""
    total = d.year * 12 + (d.month - 1) + months
    return dt.date(total // 12, total % 12 + 1, 1)


def _iter_date_range(start: dt.date, end: dt.date, window_days: int | None = None):
    """將 [start, end] 切成多個不超過 WINDOW_DAYS / window_days 嘅區間。

    預設對齊月頭切 3 個日曆月（最多 92 日，實測安全）。
    """
    if window_days:
        cur = start
        while cur <= end:
            nxt = min(cur + dt.timedelta(days=window_days - 1), end)
            yield cur, nxt
            cur = nxt + dt.timedelta(days=1)
        return

    cur = start.replace(day=1)
    while cur <= end:
        nxt = _add_months(cur, WINDOW_MONTHS)
        ws = max(cur, start)
        we = min(nxt - dt.timedelta(days=1), end)
        if ws <= we:
            yield ws, we
        cur = nxt


# ---------------------------------------------------------------- HTTP 層


def build_payload(
    last_n: int | None = None,
    draw_type: str = "All",
    start_date: dt.date | str | None = None,
    end_date: dt.date | str | None = None,
) -> dict:
    """組出 GraphQL payload。

    - 只用 `lastNDraw`：兩個日期都傳 `None`
    - 只用日期範圍：`lastNDraw` 傳 `None`（API 會以日期範圍為準）
    """
    if isinstance(start_date, dt.date):
        start_date = _ymd(start_date)
    if isinstance(end_date, dt.date):
        end_date = _ymd(end_date)

    return {
        "operationName": "marksixResult",
        "variables": {
            "lastNDraw": clamp_last_n(last_n) if (start_date is None and end_date is None) else None,
            "startDate": start_date,
            "endDate": end_date,
            "drawType": normalize_draw_type(draw_type),
        },
        "query": MARKIX_RESULT_QUERY,
    }


def _decode_body(resp: requests.Response) -> bytes:
    """回應可能係 gzip；requests 一般會自動解，這裡再做一層保險。"""
    raw = resp.content
    looks_gzip = raw[:2] == b"\x1f\x8b"
    if looks_gzip or resp.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw


def _post_graphql(payload: dict, timeout: int = DEFAULT_TIMEOUT):
    """POST GraphQL，處理 gzip、重試、GraphQL errors。

    回傳：
      * `list`  — 成功（可能係空陣列，代表該條件冇資料）
      * `None`  — 請求失敗（網絡／HTTP／GraphQL error）
    """
    last_err: str = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=timeout,
            )
        except requests.RequestException as e:
            last_err = f"請求失敗: {e}"
            print(f"  [!] {last_err}（第 {attempt}/{MAX_RETRIES} 次）", file=sys.stderr)
            time.sleep(1.5 * attempt)
            continue

        body_bytes = _decode_body(resp)

        if resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}"
            print(f"  [!] {last_err}（第 {attempt}/{MAX_RETRIES} 次）", file=sys.stderr)
            time.sleep(1.5 * attempt)
            continue

        if resp.status_code == 400:
            # 例：drawType 值錯 -> "Your input doesn't match the data type."
            print(
                f"  [!] HTTP 400（輸入唔符合 schema，通常係 drawType 大小寫）："
                f"{body_bytes.decode('utf-8', 'replace')[:160]}",
                file=sys.stderr,
            )
            return None

        if resp.status_code != 200:
            print(f"  [!] HTTP {resp.status_code}，放棄。", file=sys.stderr)
            return None

        try:
            body = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except ValueError:
            print(f"  [!] 非 JSON 回應（{len(resp.content)} bytes）", file=sys.stderr)
            return None

        if body.get("errors"):
            print(f"  [!] GraphQL errors: {body['errors']}", file=sys.stderr)
            # WHITELIST_ERROR 代表 query 被改動，重試無意義
            return None

        draws = (body.get("data") or {}).get("lotteryDraws") or []
        if not isinstance(draws, list):
            print("  [!] 回應結構異常：lotteryDraws 不是陣列", file=sys.stderr)
            return None
        return draws

    print(f"  [!] 重試 {MAX_RETRIES} 次仍失敗：{last_err}", file=sys.stderr)
    return None


# ---------------------------------------------------------------- 抓取


def fetch_draws(
    last_n: int = DEFAULT_LAST_N_DRAW,
    draw_type: str = "All",
    delay: float = DEFAULT_DELAY,
) -> list[dict]:
    """`lastNDraw` 模式：抓最近 last_n 期（raw 記錄，最新排最前）。

    ⚠️ 實測 >58 會靜默截斷；此模式回傳嘅組合可能唔連續（要完整連續資料請用
    `fetch_draws_range`）。失敗回 `[]`。
    """
    draws = _post_graphql(build_payload(last_n=last_n, draw_type=draw_type))
    if delay:
        time.sleep(delay)
    return draws or []


def fetch_draws_range(
    start: dt.date | str,
    end: dt.date | str,
    draw_type: str = "All",
    delay: float = DEFAULT_DELAY,
) -> list[dict]:
    """日期範圍模式（**完整且連續**）：`YYYYMMDD` 無 dash。

    窗口必須 ≤ 約 3 個月；超過會靜默回空陣列（呼叫方需自行處理）。
    """
    draws = _post_graphql(
        build_payload(start_date=start, end_date=end, draw_type=draw_type)
    )
    if delay:
        time.sleep(delay)
    return draws or []


def _fetch_window(
    ws: dt.date,
    we: dt.date,
    draw_type: str = "All",
    delay: float = DEFAULT_DELAY,
) -> tuple[list[dict], str]:
    """抓一個窗口，自動應對「窗口過大 → 靜默回空」。

    回傳 (draws, status)：
      * status="ok"    有資料
      * status="gap"   確認係真實資料缺口（已縮到 <= MIN_WINDOW_DAYS 仍空）
      * status="error" 請求失敗（draws 為 []）
    """
    # 窗口過大：切兩半再抓（唔可以截斷，否則會靜默漏掉尾段資料）
    if (we - ws).days > MAX_WINDOW_SPAN_DAYS:
        mid = ws + dt.timedelta(days=MAX_WINDOW_SPAN_DAYS)
        left, ls = _fetch_window(ws, mid, draw_type, delay)
        if ls == "error":
            return [], "error"
        right, rs = _fetch_window(mid + dt.timedelta(days=1), we, draw_type, delay)
        if rs == "error":
            return [], "error"
        if not left and not right:
            return [], "gap"
        return left + right, "ok"

    draws = fetch_draws_range(ws, we, draw_type=draw_type, delay=delay)
    if draws:
        return draws, "ok"

    # 空：可能係 (a) 窗口過大 (b) 真實缺口。用二分法分辨。
    if (we - ws).days <= MIN_WINDOW_DAYS:
        return [], "gap"

    mid = ws + dt.timedelta(days=(we - ws).days // 2)
    left, ls = _fetch_window(ws, mid, draw_type, delay)
    if ls == "error":
        return [], "error"
    right, rs = _fetch_window(mid + dt.timedelta(days=1), we, draw_type, delay)
    if rs == "error":
        return [], "error"
    if not left and not right:
        return [], "gap"
    return left + right, "ok"


# ---------------------------------------------------------------- 正規化


def _to_float(v):
    """把 '12493490' / '1,234' / '' / None 轉成 float 或 None。"""
    if v is None or v == "" or v == "-":
        return None
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _to_int(v):
    f = _to_float(v)
    return int(f) if f is not None else None


def _iso_to_ddmmyyyy(value: str | None) -> str | None:
    """'2026-09-19+08:00' / '2026-09-19T21:15:00+08:00' -> '19/09/2026'。"""
    if not value:
        return None
    head = str(value).strip().split("+")[0].split("T")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(head, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None


def is_result(rec: dict) -> bool:
    """只有 `status == "Result"` 且已出開獎號碼，才算已開獎。"""
    if (rec.get("status") or "") != "Result":
        return False
    dr = rec.get("drawResult") or {}
    return bool(dr.get("drawnNo"))


def normalize(rec: dict) -> dict:
    """將 GraphQL `LotteryDraw` 正規化為統一主表欄位（見 build_history.COLUMNS）。"""
    pool = rec.get("lotteryPool") or {}
    draw = rec.get("drawResult") or {}

    numbers = sorted(int(x) for x in (draw.get("drawnNo") or []) if str(x).strip() != "")
    extra = _to_int(draw.get("xDrawnNo"))

    prizes: dict[int, dict] = {}
    for p in pool.get("lotteryPrizes") or []:
        t = _to_int(p.get("type"))
        if t is not None:
            prizes[t] = p

    out = {
        "draw_id": rec.get("id"),
        "year": rec.get("year"),
        "no": _to_int(rec.get("no")),
        "date": _iso_to_ddmmyyyy(rec.get("drawDate")),  # DD/MM/YYYY（配合分析流程）
        "draw_date_iso": (rec.get("drawDate") or "").split("+")[0] or None,
        "numbers": numbers,
        "extra_ball": extra,
        "snowball_code": rec.get("snowballCode") or "",
        "snowball_name_en": rec.get("snowballName_en") or "",
        "snowball_name_ch": rec.get("snowballName_ch") or "",
        "status": rec.get("status"),
        "total_turnover": _to_float(pool.get("totalInvestment")),
        "jackpot": _to_float(pool.get("jackpot")),
        "unit_bet": _to_int(pool.get("unitBet")),
        "sell": pool.get("sell"),
    }

    for t in range(1, 8):
        p = prizes.get(t, {})
        out[f"p{t}"] = _to_float(p.get("dividend"))
        out[f"p{t}_units"] = _to_int(p.get("winningUnit"))

    return out


def fetch_normalized(
    last_n: int = DEFAULT_LAST_N_DRAW,
    draw_type: str = "All",
    delay: float = DEFAULT_DELAY,
) -> list[dict]:
    """抓最近 N 期 + 過濾已開獎 + 正規化，按 (year, no) 升序。"""
    recs = [normalize(d) for d in fetch_draws(last_n, draw_type, delay) if is_result(d) and d.get("id")]
    recs.sort(key=lambda r: (str(r.get("year") or ""), r.get("no") or 0))
    return recs


# ---------------------------------------------------------------- 寫檔


def _write_segment(out_dir: Path, name: str, recs: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = out_dir / name
    seg.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return seg


def _seg_name(ws: dt.date, we: dt.date) -> str:
    return f"seg_{ws.strftime('%Y%m%d')}_to_{we.strftime('%Y%m%d')}.json"


# ---------------------------------------------------------------- 增量：上次數據到今日


def _ddmmyyyy_to_date(value: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(str(value).strip(), "%d/%m/%Y").date()
    except (ValueError, TypeError):
        return None


def last_stored_date(
    master_csv: Path | None = None,
    out_dir: Path | None = None,
) -> dt.date | None:
    """判斷「對上一次已儲存嘅數據」係邊一日（最大開獎日期）。

    優先順序：
      1. 歷史主表 `data/mark6_history.csv`（date 欄位格式 `DD/MM/YYYY`）
      2. 冇主表就掃 `data/raw/*.json` 內嘅 `drawDate`（ISO 格式）
      3. 都冇 → `None`
    """
    master = Path(master_csv) if master_csv else HISTORY_CSV
    if master.exists():
        try:
            import csv

            with master.open(encoding="utf-8") as f:
                dates = [
                    d for d in (_ddmmyyyy_to_date(row.get("date", "")) for row in csv.DictReader(f))
                    if d
                ]
            if dates:
                return max(dates)
        except (OSError, ValueError):
            pass

    # 退回掃 raw JSON 快取
    raw = Path(out_dir) if out_dir else RAW_DIR
    if not raw.exists():
        return None
    found: list[dt.date] = []
    for path in sorted(raw.glob("*.json")):
        try:
            for rec in json.loads(path.read_text(encoding="utf-8")):
                iso = (rec.get("drawDate") or "").split("+")[0].split("T")[0]
                if len(iso) >= 10:
                    try:
                        found.append(dt.date.fromisoformat(iso[:10]))
                    except ValueError:
                        continue
        except (OSError, ValueError):
            continue
    return max(found) if found else None


def fetch_since_last(
    out_dir: Path | None = None,
    draw_type: str = DRAW_TYPE_ALL,
    delay: float = DEFAULT_DELAY,
    today: dt.date | None = None,
    master_csv: Path | None = None,
) -> dict:
    """增量更新：由「最後一期已儲存數據」抓到今日。

    流程：
      1. 用 `last_stored_date()` 找對上一次數據日期（主表 → raw 快取）
      2. 由該日（含，重抓最後一期以防之前未開獎／有更正）抓至今
      3. 寫入 `data/raw/last_all.json`（**每次覆寫，唔會碎片化**）
      4. 完全冇舊數據 → 改為 `fetch_history()` 由 1993 完整抓取（可續傳）

    `build_history --mode build` 會同時讀 seg_*.json 及 last_all.json，
    按 draw_id 去重，所以重疊資料無害。
    """
    out_dir = Path(out_dir) if out_dir else RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    today = today or dt.date.today()
    last = last_stored_date(master_csv=master_csv, out_dir=out_dir)

    if last is None:
        print(
            "[GraphQL] 主表／raw 快取都冇資料 → 由 1993 開始做完整歷史抓取"
            "（約 3-4 分鐘，分段寫入、可中斷續傳）。"
        )
        draws = fetch_history(
            EARLIEST_DATE, today, out_dir, draw_type=draw_type, delay=delay
        )
        return {
            "last_stored": None,
            "start": EARLIEST_DATE.isoformat(),
            "end": today.isoformat(),
            "windows": len(list(_iter_date_range(EARLIEST_DATE, today))),
            "draws": draws,
            "incremental": False,
        }

    start = min(last, today)
    windows = list(_iter_date_range(start, today))
    print(
        f"[GraphQL] 對上一次已儲存數據：{last} → 由 {start} 抓到 {today}"
        f"（{(today - start).days + 1} 日、{len(windows)} 個窗口）"
    )

    collected: list[dict] = []
    for i, (ws, we) in enumerate(windows, 1):
        draws, status = _fetch_window(ws, we, draw_type=draw_type, delay=delay)
        if status == "error":
            print(f"  [!] 窗口 {ws} ~ {we} 請求失敗。", file=sys.stderr)
            if not collected:
                return {
                    "last_stored": last.isoformat(), "start": start.isoformat(),
                    "end": today.isoformat(), "windows": len(windows),
                    "draws": 0, "incremental": True, "error": True,
                }
            break
        label = "  （此區間無攪珠）" if status == "gap" else ""
        print(f"  [{i}/{len(windows)}] {ws} ~ {we}: {len(draws):>3} 筆{label}")
        collected.extend(draws)

    # 去重（同一期可能因重疊而重覆）
    uniq: dict[str, dict] = {}
    for d in collected:
        if d.get("id"):
            uniq[d["id"]] = d
    ordered = sorted(
        uniq.values(),
        key=lambda d: (str(d.get("year") or ""), d.get("no") or 0),
        reverse=True,
    )
    seg = _write_segment(out_dir, "last_all.json", ordered)
    new_results = [d for d in ordered if is_result(d)]
    print(f"  寫入 {len(ordered)} 筆（已開獎 {len(new_results)} 筆）-> {seg.name}")

    return {
        "last_stored": last.isoformat(),
        "start": start.isoformat(),
        "end": today.isoformat(),
        "windows": len(windows),
        "draws": len(ordered),
        "incremental": True,
    }


def fetch_last_n(
    last_n: int = DEFAULT_LAST_N_DRAW,
    out_dir: Path = Path("data/raw"),
    draw_type: str = "All",
    delay: float = DEFAULT_DELAY,
) -> int:
    """`lastNDraw` 模式：抓最近 last_n 期，raw 記錄寫入 `last_all.json`。"""
    n = clamp_last_n(last_n)
    print(f"[GraphQL] 抓取最近 {n} 期（drawType={normalize_draw_type(draw_type)}）...")
    draws = fetch_draws(n, draw_type=draw_type, delay=delay)
    if not draws:
        print("  [!] 冇取得任何資料。", file=sys.stderr)
        return 0

    seg = _write_segment(Path(out_dir), "last_all.json", draws)
    results = [d for d in draws if is_result(d)]
    span = ""
    if results:
        ds = sorted((d.get("drawDate") or "")[:10] for d in results if d.get("drawDate"))
        if ds:
            span = f"，日期 {ds[0]} ~ {ds[-1]}"
    print(f"完成，共 {len(draws)} 筆（已開獎 {len(results)} 筆{span}）-> {seg.name}")
    return len(draws)


def fetch_latest(
    out_dir: Path,
    delay: float = DEFAULT_DELAY,
    days: int = 60,
    draw_type: str = DRAW_TYPE_ALL,
) -> int:
    """增量更新：用**日期範圍**抓最近 days 日（完整連續），寫入 `last_all.json`。"""
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    print(f"[GraphQL] 增量抓取 {start} ~ {end}（drawType={normalize_draw_type(draw_type)}）...")
    draws, status = _fetch_window(start, end, draw_type=draw_type, delay=delay)
    if status == "error":
        print("  [!] 抓取失敗。", file=sys.stderr)
        return 0

    seg = _write_segment(Path(out_dir), "last_all.json", draws)
    results = [d for d in draws if is_result(d)]
    print(
        f"完成，共 {len(draws)} 筆（已開獎 {len(results)} 筆）"
        f"{'  [注意：此區間無攪珠記錄]' if not draws else ''} -> {seg.name}"
    )
    return len(draws)


def fetch_history(
    start: dt.date,
    end: dt.date,
    out_dir: Path,
    draw_type: str = DRAW_TYPE_ALL,
    delay: float = DEFAULT_DELAY,
    resume: bool = True,
    force: bool = False,
) -> int:
    """完整歷史抓取：由 start 到 end，逐個 ≤3 個月窗口寫入 `seg_*.json`。

    - 使用日期範圍模式（`YYYYMMDD`），資料完整且連續（**與 lastNDraw 唔同**）
    - 窗口過大會靜默回空 → 自動二分縮窗重試
    - `resume=True`：已存在嘅 seg 檔會跳過（斷點續傳）；`force=True` 強制重抓
    - 回傳寫入／命中嘅去重 draw_id 數目
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if start < EARLIEST_DATE:
        print(f"  [!] HKJC 最早資料為 {EARLIEST_DATE}（1993-01-05 第一期），已由該日起。")
        start = EARLIEST_DATE
    if end < start:
        print("  [!] 結束日期早於開始日期。", file=sys.stderr)
        return 0

    windows = list(_iter_date_range(start, end))
    print(
        f"[GraphQL] 完整歷史：{start} ~ {end}，共 {len(windows)} 個窗口"
        f"（每窗口最多 {WINDOW_MONTHS} 個月，drawType={normalize_draw_type(draw_type)}）"
    )

    seen: set[str] = set()
    gaps: list[str] = []
    cached = 0
    for i, (ws, we) in enumerate(windows, 1):
        seg = out_dir / _seg_name(ws, we)
        if resume and not force and seg.exists():
            try:
                for r in json.loads(seg.read_text(encoding="utf-8")):
                    if r.get("id"):
                        seen.add(r["id"])
                cached += 1
                continue
            except (ValueError, OSError):
                pass  # 壞檔 → 重抓

        draws, status = _fetch_window(ws, we, draw_type=draw_type, delay=delay)
        if status == "error":
            print(f"  [!] 窗口 {ws} ~ {we} 請求失敗，中止（下次執行會續傳）。", file=sys.stderr)
            break

        _write_segment(out_dir, seg.name, draws)
        for d in draws:
            if d.get("id"):
                seen.add(d["id"])

        if status == "gap":
            gaps.append(f"{ws}~{we}")
            print(f"  [{i}/{len(windows)}] {ws} ~ {we}: 0 筆  (確認無攪珠記錄)")
        else:
            print(f"  [{i}/{len(windows)}] {ws} ~ {we}: {len(draws):>3} 筆 -> {seg.name}")

    print(
        f"完成：唯一期數 {len(seen)} 筆"
        + (f"（沿用舊檔 {cached} 個窗口）" if cached else "")
    )
    if gaps:
        print(f"     無攪珠記錄嘅窗口 {len(gaps)} 個：" + ", ".join(gaps[:6]) + (" …" if len(gaps) > 6 else ""))
    return len(seen)


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="抓取香港賽馬會六合彩開獎記錄（GraphQL；lastNDraw 最多 58 期，或按日期範圍）"
    )
    p.add_argument("--from", dest="start", help="起始日期 YYYY-MM-DD（預設：--latest 以外為 1993-01-01）")
    p.add_argument("--to", dest="end", help="結束日期 YYYY-MM-DD（預設今日）")
    p.add_argument("--out", default="data/raw", help="原始 JSON 輸出目錄（預設 data/raw）")
    p.add_argument(
        "--latest",
        action="store_true",
        help="增量更新：由最後一期已儲存數據抓到今日（同 --since-last）",
    )
    p.add_argument(
        "--since-last",
        dest="since_last",
        action="store_true",
        help="增量更新：自動判斷上次數據日期，抓到今日",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=None,
        help=f"用 lastNDraw 抓最近 N 期（預設 {DEFAULT_LAST_N_DRAW}，硬上限 {MAX_LAST_N_DRAW}）",
    )
    p.add_argument("--draw-type", default=DRAW_TYPE_ALL, choices=DRAW_TYPES, help="All 或 SnowBall（預設 All）")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="每次請求延遲秒數")
    p.add_argument("--days", type=int, help="只用固定回溯日數（覆寫 --latest/--since-last 嘅自動判斷）")
    p.add_argument("--force", action="store_true", help="忽略已存在嘅分段檔，強制重抓")
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    a = parse_args(argv)
    out_dir = Path(a.out)

    if a.days is not None:
        # 明確指定回溯日數
        fetch_latest(out_dir, delay=a.delay, days=a.days, draw_type=a.draw_type)
        return

    if a.latest or a.since_last:
        # 自動判斷「對上一次數據」係邊日，然後抓到今日
        stats = fetch_since_last(out_dir, draw_type=a.draw_type, delay=a.delay)
        if stats.get("error"):
            print("[!] 增量抓取失敗。", file=sys.stderr)
            return
        if stats["incremental"]:
            print(
                f"完成：上次數據 {stats['last_stored']} → 抓到 {stats['end']}，"
                f"共取得 {stats['draws']} 筆（新期數會由 build_history 去重後追加）。"
            )
        else:
            print(f"完成：完整抓取 {stats['start']} ~ {stats['end']}，共 {stats['draws']} 期。")
        return

    if a.start or a.end:
        # 日期範圍模式 → 完整歷史（可斷點續傳）
        start = dt.date.fromisoformat(a.start) if a.start else EARLIEST_DATE
        end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
        fetch_history(
            start, end, out_dir,
            draw_type=a.draw_type, delay=a.delay, force=a.force,
        )
        return

    last_n = a.last_n if a.last_n is not None else DEFAULT_LAST_N_DRAW
    fetch_last_n(last_n=last_n, out_dir=out_dir, draw_type=a.draw_type, delay=a.delay)


if __name__ == "__main__":
    main()
