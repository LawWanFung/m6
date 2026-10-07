"""數據管線迴歸測試（離線；網絡調用全部 monkeypatch，不發任何請求）。

覆蓋的迴歸（對應 docs/BUG_TODO.md）：
  1. analyze.load：壞行/空號碼/壞日期 → 跳過並計數，兩種日期格式都支持
  2. chi_square_uniform：自由度 = 參與檢驗的類別數 - 1（舊版用 pool 大小 - 1）
  3. build_history：主表為底 + raw 疊合；Pending 不入表；按日期排序；
     逐字段合併（空值不覆蓋更完整的舊記錄）
  4. hkjc_fetch 三態：請求失敗 ≠ 無數據
     - 失敗 → 不寫空分段檔、不覆寫 last_all.json、CLI exit 1（下次重抓）
     - 真實缺口 → 寫空分段檔（避免重複抓取）
  5. CLI --since-last 在請求失敗時返回 1（run_all --latest 依賴此傳播失敗）
  6. update_latest：補錄舊日期攪珠仍保持整檔日期升序；header 唔一致 → 中止
  7. fetch_history 反向區間 → 回 (0, []) 元組（唔係裸 int）
  8. _post_graphql 對 429 退避重試（唔再一 429 就終態失敗）
  9. _load_seg 接納已正規化嘅記錄（唔二次 normalize）
 10. analyze._write_analysis：內容未變 → 唔重寫（唔推 mtime）
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import analyze.frequency as freqmod
from src import build_history as bh
from src import hkjc_fetch as hf


# ───────────────────────── 1. analyze.load ─────────────────────────


def test_load_skips_bad_rows_counts_bad_dates(make_csv):
    rows = [
        ["OK1", "01/01/2020", "1,2,3,4,5,6", "7"],
        ["BAD1", "02/01/2020", "", "8"],                     # 空號碼
        ["BAD2", "03/01/2020", "1,2,3,4,5", "9"],           # 5 個號碼
        ["OK2", "2020-02-01", "10,11,12,13,14,15", "16"],   # ISO 日期
        ["BAD3", "notadate", "20,21,22,23,24,25", "26"],    # 壞日期（保留但 NaT）
    ]
    df = freqmod.load(make_csv(rows))
    assert len(df) == 3                       # 2 行壞號碼被丟棄；壞日期行保留
    assert df.attrs["dropped_rows"] == 2
    assert df.attrs["bad_dates"] == 1
    assert df.attrs["data_source"] == "real"
    # ISO 日期應被正確解析
    assert df["date"].tolist() == [
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.NaT
    ]


def test_load_refuses_sample_unless_allowed(tmp_path):
    import analyze.frequency as fm
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        fm.load(missing)                      # 默認絕不退回合成樣本
    if (fm.SAMPLE_CSV).exists():
        df = fm.load(missing, allow_sample=True)
        assert df.attrs["data_source"] == "sample"


def test_chi_square_dof_and_empty_guard():
    # 約定：調用方先把 freq reindex 到該時代的 pool；這裡傳滿 pool 的 series
    res = freqmod.chi_square_uniform(pd.Series([100] * 49), "all", total=4900)
    assert res["dof"] == 48 and res["n_categories"] == 49
    assert res["chi2"] < 1e-9 and res["p_value"] > 0.99   # 完美均勻

    # 偏斜分佈 → chi2 增大
    res = freqmod.chi_square_uniform(pd.Series([4900] + [0] * 48), "skew", total=4900)
    assert res["chi2"] > 1000 and res["significant_05"] is True

    # 空輸入（total=0 → 期望全 0）→ 不出錯、全 None
    res = freqmod.chi_square_uniform(pd.Series([0] * 49), "empty", total=0)
    assert res["dof"] is None and res["chi2"] is None
    assert res["significant_05"] is False


def test_era_chi_reindexes_to_era_pool(tmp_path, make_csv):
    """分時代卡方：pool 由調用方 reindex（45/47/49），自由度 = pool - 1。"""
    # 一期 1993 年的數據（45 號 pool 時代）
    rows = [["X1", "05/01/1993", "1,2,3,4,5,6", "7"],
            ["X2", "12/01/1993", "1,3,5,7,9,11", "13"]]
    df = freqmod.load(make_csv(rows))
    res = freqmod._era_main_chi(df, 45, "1993-1996")
    assert res["dof"] == 44


# ───────────────────────── 2. build_history ─────────────────────────


def _master_row(did: str, date: str, numbers: str, turnover: float | None = None,
                p1: float | None = None) -> dict:
    r = {c: None for c in bh.COLUMNS}
    r.update({"draw_id": did, "date": date, "numbers": numbers, "extra_ball": 1,
              "snowball_code": "MK6", "snowball_name_ch": "六合彩"})
    if turnover is not None:
        r["total_turnover"] = turnover
    if p1 is not None:
        r["p1"] = p1
    return r


def _write_master(path: Path, rows: list[dict]) -> None:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=bh.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _raw_rec(rid, date_iso, drawn, status="Result", turnover=None):
    """raw 分段檔裡的 GraphQL 原始記錄（normalize 之前的形狀）。"""
    rec = {
        "id": rid, "year": date_iso[:4], "no": 1,
        "drawDate": date_iso + "+08:00", "status": status,
        "snowballCode": "MK6", "snowballName_en": "Mark Six", "snowballName_ch": "六合彩",
        "drawResult": {"drawnNo": list(drawn), "xDrawnNo": "49"},
    }
    if turnover is not None:
        rec["lotteryPool"] = {"totalInvestment": turnover, "lotteryPrizes": []}
    return rec


def test_build_history_merges_sorts_filters_and_preserves(tmp_path, monkeypatch):
    master = tmp_path / "data" / "mark6_history.csv"
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", raw)

    # 現有主表：A 有 turnover，B/C 最簡
    _write_master(master, [
        _master_row("A", "05/01/2020", "1,2,3,4,5,6", turnover=1000.0, p1=20000.0),
        _master_row("B", "10/01/2020", "7,8,9,10,11,12"),
        _master_row("C", "12/01/2020", "13,14,15,16,17,18"),
    ])

    # raw 分段檔：
    #  - A_dup：號碼相同、turnover 為空 → 不能覆蓋主表已有的 1000
    #  - B_rich：同一期但帶 turnover → 應補進主表
    #  - P：Pending 未開獎 → 絕不能入主表
    #  - D：新期，日期在 A、B 之間 → 應插入並按日期排序
    (raw / "seg_20200101_to_20200131.json").write_text(json.dumps([
        _raw_rec("A", "2020-01-05", ["1", "2", "3", "4", "5", "6"]),
        _raw_rec("B", "2020-01-10", ["7", "8", "9", "10", "11", "12"], turnover=9999.0),
        _raw_rec("P", "2020-01-11", [], status="Pending"),
        _raw_rec("D", "2020-01-06", ["19", "20", "21", "22", "23", "24"]),
    ], ensure_ascii=False), encoding="utf-8")

    n = bh.build_history()
    assert n == 4

    out = pd.read_csv(master)
    assert out["draw_id"].tolist() == ["A", "D", "B", "C"], "必須按真實日期升序"
    a = out[out.draw_id == "A"].iloc[0]
    assert a.total_turnover == 1000.0, "空值不能覆蓋已有的完整記錄"
    b = out[out.draw_id == "B"].iloc[0]
    assert b.total_turnover == 9999.0, "raw 更完整的字段應補進主表"
    assert "P" not in set(out.draw_id), "Pending（未開獎）不能入主表"

    # 主表日期單調不減
    parts = out["date"].str.split("/")
    key = parts.apply(lambda s: int(s[2]) * 10000 + int(s[1]) * 100 + int(s[0]))
    assert (key.diff().dropna() >= 0).all()


def test_build_history_without_raw_keeps_existing(tmp_path, monkeypatch):
    """新 clone / Docker 環境：raw 目錄不存在 → 保留現有主表，不清空。"""
    master = tmp_path / "data" / "mark6_history.csv"
    _write_master(master, [
        _master_row("A", "05/01/2020", "1,2,3,4,5,6"),
        _master_row("B", "10/01/2020", "7,8,9,10,11,12"),
    ])
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", tmp_path / "data" / "raw_missing")
    assert bh.build_history() == 2
    assert len(pd.read_csv(master)) == 2


# ───────────────────────── 3. hkjc_fetch 三態 ─────────────────────────


def test_fetch_last_n_failure_never_overwrites_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_post_graphql", lambda payload, timeout=60: ([], "error"))
    out = tmp_path / "raw"
    out.mkdir()
    seg = out / "last_all.json"
    seg.write_text(json.dumps([{"id": "keep-me"}]), encoding="utf-8")

    assert hf.fetch_last_n(last_n=10, out_dir=out, delay=0) == -1
    assert json.loads(seg.read_text(encoding="utf-8")) == [{"id": "keep-me"}], \
        "請求失敗不能覆寫已累積的 raw 快取"


def test_fetch_last_n_empty_response_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_post_graphql", lambda payload, timeout=60: ([], "empty"))
    out = tmp_path / "raw"
    out.mkdir()
    seg = out / "last_all.json"
    seg.write_text(json.dumps([{"id": "keep-me"}]), encoding="utf-8")

    assert hf.fetch_last_n(last_n=10, out_dir=out, delay=0) == 0
    assert json.loads(seg.read_text(encoding="utf-8")) == [{"id": "keep-me"}]


def test_fetch_history_error_writes_no_seg_file(tmp_path, monkeypatch):
    """請求失敗的窗口：不寫分段檔（否則 resume 會永遠跳過 → 永久缺口）。"""
    calls = []

    def fake_post(payload, timeout=60):
        calls.append(payload)
        return [], "error"

    monkeypatch.setattr(hf, "_post_graphql", fake_post)
    out = tmp_path / "raw"
    out.mkdir()
    count, errors = hf.fetch_history(
        dt.date(2020, 1, 1), dt.date(2020, 1, 3), out, delay=0)

    assert count == 0
    assert len(errors) == 1
    assert calls, "必須真的嘗試請求"
    assert list(out.glob("*.json")) == [], "失敗窗口不能留下分段檔"


def test_fetch_history_gap_writes_empty_seg(tmp_path, monkeypatch):
    """確認的真實缺口（200 + 空 + 已縮到最小窗口）：寫空分段檔，避免重複抓取。"""
    monkeypatch.setattr(hf, "_post_graphql", lambda payload, timeout=60: ([], "empty"))
    out = tmp_path / "raw"
    out.mkdir()
    count, errors = hf.fetch_history(
        dt.date(2020, 1, 1), dt.date(2020, 1, 3), out, delay=0)
    assert count == 0
    assert errors == []
    assert list(out.glob("seg_*.json")), "真實缺口應寫（空）分段檔標記已確認"


def test_cli_since_last_error_exits_1(tmp_path, monkeypatch, capsys):
    """P0 迴歸：網絡失敗 → 非零 exit，不寫 last_all.json（run_all --latest 依賴）。"""
    monkeypatch.setattr(hf, "_post_graphql", lambda payload, timeout=60: ([], "error"))
    out = tmp_path / "raw"
    out.mkdir()

    rc = hf.main(["--since-last", "--out", str(out)])
    assert rc == 1, "抓取失敗必須返回非零 exit code"
    assert not (out / "last_all.json").exists(), "失敗不能寫空 last_all.json"


def test_cli_since_last_success_exits_0(tmp_path, monkeypatch):
    rec = _raw_rec("X1", "2026-10-03", ["1", "2", "3", "4", "5", "6"])

    def fake_post(payload, timeout=60):
        return [rec], "ok"

    monkeypatch.setattr(hf, "_post_graphql", fake_post)
    out = tmp_path / "raw"
    out.mkdir()
    rc = hf.main(["--since-last", "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "last_all.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["X1"]


def test_fetch_history_reversed_range_returns_tuple(tmp_path, monkeypatch):
    """R2-1：start > end → 回 (0, []) 元組（舊版回裸 0 → CLI 解包 TypeError）。"""
    count, errors = hf.fetch_history(
        dt.date(2020, 1, 10), dt.date(2020, 1, 1), tmp_path / "raw", delay=0)
    assert count == 0 and errors == []


def test_post_graphql_retries_429_then_succeeds(monkeypatch, capsys):
    """R2-15：429 = 瞬態（退避重試），唔係終態。"""
    import io

    class _Resp:
        def __init__(self, code, body=b"{}"): self.status_code, self._b = code, body
        @property
        def content(self): return self._b
        headers = {}

    calls = {"n": 0}

    class _FakeSession:
        def post(self, url, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(429)
            return _Resp(200, json.dumps({
                "data": {"lotteryDraws": [_raw_rec("X1", "2026-10-03", ["1","2","3","4","5","6"])]}
            }).encode())

    monkeypatch.setattr(hf, "_session", lambda: _FakeSession())
    monkeypatch.setattr(hf.time, "sleep", lambda s: None)
    draws, status = hf._post_graphql({"variables": {}})
    assert status == "ok" and len(draws) == 1
    assert calls["n"] == 2, "第一次 429 後必須重試"


# ────────────── 6. update_latest（增量合併）─────────────


def test_update_latest_backfill_keeps_file_sorted(tmp_path, monkeypatch):
    """R2-6：補錄嘅舊日期攪珠（Pending 遲先開獎）必須插入正確位置，
    唔係 append 到尾部打亂時序（model/base.py 窗口特徵按行序取）。
    """
    master = tmp_path / "data" / "mark6_history.csv"
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", tmp_path / "raw")  # 隔離開發機嘅真實 raw 快取
    _write_master(master, [
        _master_row("A", "05/01/2020", "1,2,3,4,5,6"),
        _master_row("B", "10/01/2020", "7,8,9,10,11,12"),
        _master_row("C", "12/01/2020", "13,14,15,16,17,18"),
    ])
    # 抓回嚟：一條**舊於**現有尾段嘅補錄（07/01）+ 一條新期（15/01）
    monkeypatch.setattr(bh, "_fetch_range", lambda last_n, delay: (
        [bh.normalize(_raw_rec("BACK", "2020-01-07", ["20","21","22","23","24","25"])),
         bh.normalize(_raw_rec("NEW", "2020-01-15", ["26","27","28","29","30","31"]))],
        "ok"))

    added = bh.update_latest()
    assert added == 2
    out = pd.read_csv(master)
    assert out["draw_id"].tolist() == ["A", "BACK", "B", "C", "NEW"], \
        "整檔必須日期升序（補錄插入中間，唔係堆尾）"
    assert out["date"].tolist() == ["05/01/2020", "07/01/2020", "10/01/2020",
                                    "12/01/2020", "15/01/2020"]


def test_update_latest_header_mismatch_aborts(tmp_path, monkeypatch):
    """R2-6：舊 schema 主表（header 唔同 COLUMNS）→ 中止，唔改檔案（防字段漂移）。"""
    master = tmp_path / "data" / "mark6_history.csv"
    master.parent.mkdir(parents=True)
    master.write_text("draw_id,date,nums,xb\nA,05/01/2020,1,2,3,4,5,6,7\n", encoding="utf-8")
    before = master.read_text(encoding="utf-8")
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", tmp_path / "raw")  # 隔離開發機嘅真實 raw 快取
    monkeypatch.setattr(bh, "_fetch_range", lambda last_n, delay: (
        [bh.normalize(_raw_rec("X1", "2026-10-03", ["1","2","3","4","5","6"]))], "ok"))
    assert bh.update_latest() == -1
    assert master.read_text(encoding="utf-8") == before, "檔案必須原封不動"


def test_update_latest_failure_leaves_master_untouched(tmp_path, monkeypatch):
    master = tmp_path / "data" / "mark6_history.csv"
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", tmp_path / "raw")  # 隔離開發機嘅真實 raw 快取
    _write_master(master, [_master_row("A", "05/01/2020", "1,2,3,4,5,6")])
    before = master.read_text(encoding="utf-8")
    monkeypatch.setattr(bh, "_fetch_range", lambda last_n, delay: ([], "error"))
    assert bh.update_latest() == -1
    assert master.read_text(encoding="utf-8") == before


def test_load_seg_accepts_already_normalized(tmp_path):
    """R2-17：已正規化嘅記錄（有 draw_id、無 GraphQL id）唔二次 normalize（會丟 numbers）。"""
    p = tmp_path / "seg_norm.json"
    p.write_text(json.dumps([{
        "draw_id": "N1", "date": "05/01/2020", "numbers": [1, 2, 3, 4, 5, 6],
        "extra_ball": 7, "status": "Result",
    }]), encoding="utf-8")
    recs = bh._load_seg(p)
    assert len(recs) == 1 and recs[0]["numbers"] == [1, 2, 3, 4, 5, 6]


def test_write_analysis_skips_when_unchanged(tmp_path, monkeypatch):
    """R2-10：內容未變 → 唔重寫（唔推 mtime → /api/* 緩存唔會無端失效）。"""
    import analyze.frequency as fm
    import time as _t

    monkeypatch.setattr(fm, "RESULTS_DIR", tmp_path)
    out = {"a": 1, "date_range": {"first": None, "last": None}}
    assert fm._write_analysis(out) is True
    mtime1 = (tmp_path / "analysis.json").stat().st_mtime_ns
    _t.sleep(0.01)
    assert fm._write_analysis(out) is False, "內容未變唔應該重寫"
    assert (tmp_path / "analysis.json").stat().st_mtime_ns == mtime1
    # 內容變了 → 要寫
    assert fm._write_analysis({"a": 2}) is True


# ───────────────────── R3（2026-10-07 部署事故） ─────────────────────


def test_update_latest_reuses_fresh_last_all_without_api(tmp_path, monkeypatch):
    """R3-1：update_latest 必須重用「新」last_all.json（上一步 fetch 剛寫咗），
    唔可以再打第二次 HKJC（實測：佢限流「靜默返空」→ 剛先抓到嘅新攪珠被
    誤判「冇新數據」而丟棄）。
    """
    import os as _os
    import time as _t

    master = tmp_path / "data" / "mark6_history.csv"
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", raw)
    _write_master(master, [_master_row("A", "05/01/2020", "1,2,3,4,5,6")])

    # 「剛剛寫」嘅 last_all.json（raw 格式、新期）
    seg = raw / "last_all.json"
    seg.write_text(json.dumps([
        _raw_rec("NEW1", "2020-01-15", ["26", "27", "28", "29", "30", "31"]),
    ]), encoding="utf-8")
    now = _t.time()
    _os.utime(seg, (now, now))

    def _boom(last_n, delay):
        raise AssertionError("有新嘅 last_all.json 時唔應該再打 API")

    monkeypatch.setattr(bh, "_fetch_range", _boom)
    assert bh.update_latest() == 1
    out = pd.read_csv(master)
    assert out["draw_id"].tolist() == ["A", "NEW1"]


def test_update_latest_refetches_when_last_all_stale(tmp_path, monkeypatch):
    """R3-1：last_all.json 過舊（獨立手動執行情境）→ 仍然去 API 重抓。"""
    import os as _os
    import time as _t

    master = tmp_path / "data" / "mark6_history.csv"
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(bh, "HISTORY_CSV", master)
    monkeypatch.setattr(bh, "RAW_DIR", raw)
    _write_master(master, [_master_row("A", "05/01/2020", "1,2,3,4,5,6")])

    seg = raw / "last_all.json"
    seg.write_text(json.dumps([_raw_rec("X", "2020-01-15", ["1", "2", "3", "4", "5", "6"])]),
                   encoding="utf-8")
    old = _t.time() - 3600  # 一小時前（超出 600 秒新淨窗口）
    _os.utime(seg, (old, old))

    calls = []
    monkeypatch.setattr(bh, "_fetch_range",
                        lambda last_n, delay: (calls.append(1), ([], "ok"))[1])
    bh.update_latest()
    assert calls == [1], "舊快取必須重抓"


def test_fetch_window_min_empty_retries_once(monkeypatch):
    """R3-2：最小窗口返 200+空（實測：限流時唔會 429、靜默返空）→ 重試一次先斷言
    「真實缺口」。"""

    class _NoSleep:
        def sleep(self, s):
            pass

    monkeypatch.setattr(hf, "time", _NoSleep())

    calls = []

    def fake_range(ws, we, draw_type="All", delay=0.0):
        calls.append(1)
        return ([{"id": "X1"}], "ok") if len(calls) == 2 else ([], "empty")

    monkeypatch.setattr(hf, "fetch_draws_range", fake_range)

    d, s = hf._fetch_window(dt.date(2026, 9, 26), dt.date(2026, 10, 5), "All", 0.0)
    assert s == "ok" and d == [{"id": "X1"}], "重試返到資料 → ok"
    assert len(calls) == 2


def test_fetch_window_min_empty_still_gap_after_retry(monkeypatch):
    """R3-2：重試後仍然空 → 先斷言真實缺口（gap），唔係 error。"""

    class _NoSleep:
        def sleep(self, s):
            pass

    monkeypatch.setattr(hf, "time", _NoSleep())
    calls = []

    def fake_range(ws, we, draw_type="All", delay=0.0):
        calls.append(1)
        return [], "empty"

    monkeypatch.setattr(hf, "fetch_draws_range", fake_range)

    d, s = hf._fetch_window(dt.date(2026, 9, 26), dt.date(2026, 10, 5), "All", 0.0)
    assert s == "gap" and d == []
    assert len(calls) == 2
