"""
一鍵流程
========

依序執行：
  1. 抓取歷史（或增量）
  2. 整合成主表
  3. 統計分析
  4. 建模

用法：
  python run_all.py                 # 用合成樣本跑完整流程（離線可測）
  python run_all.py --real          # 真實 HKJC 數據：默認**增量**（--since-last）；失敗時中止，唔會假裝成功
  python run_all.py --real --full   #   由 1993-01-01 全量抓取（首次建立主表／重建時先要）
  python run_all.py --latest        # 只做增量更新 + 分析
  python run_all.py --real --allow-sample   # 顯式允許退回合成樣本（調試用）

⚠️ 設計規則（舊版本違反）：
  * `--real` / `--latest` 默認**不會**傳 `--allow-sample`：真實數據不可用時應該
    報錯中止，而不是靜默用合成樣本做「分析」。
  * `--real` 默認**增量**（since-last）：主表已有真實歷史，日常增量就夠；
    全量（`--full`，由 1993 起、約 26 分鐘）只留給首次／重建 —— 舊版
    `--real` 無腦全量，日常跑一次要多等半個鐘。
  * 子程序失敗（包括網絡/代理失敗、exit code != 0）必須中止並返回非零 exit code；
    唔可以繼續跑分析並回報「成功」，否則 dashboard 會顯示基於舊數據/空數據的結果。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - 有些 stdout 不可 reconfigure
    pass

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"


def run(cmd: list[str], label: str, timeout: int = 900) -> bool:
    """執行子程序；失敗或超時 → 回傳 False（唔拋 traceback）。"""
    print(f"\n===== {label} =====")
    try:
        p = subprocess.run(
            [sys.executable, *cmd],
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        print(f"[!] {label} 超時（>{timeout}s）；中止，唔會用半截數據繼續分析")
        return False
    except OSError as exc:
        print(f"[!] {label} 無法執行: {exc}", file=sys.stderr)
        return False

    if p.stdout:
        print(p.stdout.rstrip())
    if p.stderr:
        print(p.stderr.rstrip(), file=sys.stderr)

    if p.returncode != 0:
        print(f"[!] {label} 失敗（exit {p.returncode}）；中止後續步驟")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Six 一鍵流程：抓取 → 整合 → 分析 → 建模")
    ap.add_argument("--real", action="store_true",
                    help="抓取真實 HKJC 數據（默認增量；需網絡；失敗時中止）")
    ap.add_argument("--latest", action="store_true", help="只做增量更新（與 --real 默認行為相同，保留兼容）")
    ap.add_argument("--full", action="store_true",
                    help="由 1993-01-01 全量抓取（首次建立主表／重建時用；與 --latest 互斥）")
    ap.add_argument("--no-model", action="store_true", help="跳過建模")
    ap.add_argument("--allow-sample", action="store_true",
                    help="允許在無真實數據時退回合成樣本（只供離線測試/調試）")
    a = ap.parse_args()

    if a.full and a.latest:
        ap.error("--full 同 --latest 互斥")

    if a.real or a.latest:
        cmd = [str(SRC / "hkjc_fetch.py")]
        # 「增量 vs 全量」明確用 CLI 表達，默認增量（見文件頭設計規則）。
        if a.full:
            cmd += ["--from", "1993-01-01"]
        else:
            cmd.append("--since-last")
        if not run(cmd, "1) 抓取 HKJC 數據", timeout=900):
            print("[!] 抓取失敗：主表未更新。後續分析/建模用的係舊數據，結果不可信。")
            return 1
        # 真實路徑默認唔允許合成樣本；只有顯式 --allow-sample 才傳
        sample_args = ["--allow-sample"] if a.allow_sample else []
    else:
        # 離線：先生成合成樣本數據（必須明確 --allow-sample 才會用）
        if not run([str(SRC / "gen_sample.py")], "0. 生成合成樣本（離線測試）", timeout=120):
            return 1
        sample_args = ["--allow-sample"]

    if not run([str(SRC / "build_history.py"), "--mode", "update" if a.latest else "build"],
               "2) 整合成主表", timeout=300):
        return 1

    if not run([str(BASE / "analyze" / "frequency.py"), *sample_args], "3) 統計分析", timeout=300):
        return 1

    if not a.no_model:
        if not run([str(BASE / "model" / "predict.py"), "--lookback", "10", *sample_args],
                    "4) 建模", timeout=600):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
