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
  python run_all.py --real          # 用真實 HKJC 數據（需網絡可直連該網站）
  python run_all.py --latest        # 只做增量更新 + 分析
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"


def run(cmd: list[str], label: str):
    print(f"\n===== {label} =====")
    subprocess.run([sys.executable, *cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="抓取真實 HKJC 數據（需網絡可直連）")
    ap.add_argument("--latest", action="store_true", help="只做增量更新")
    ap.add_argument("--no-model", action="store_true", help="跳過建模")
    a = ap.parse_args()

    if a.real or a.latest:
        cmd = [str(SRC / "hkjc_fetch.py")]
        if a.latest:
            cmd.append("--latest")
        else:
            cmd += ["--from", "1993-01-01"]
        run(cmd, "1) 抓取 HKJC 數據")
    else:
        # 離線：先合成樣本數據，讓流程可跑
        run([str(SRC / "gen_sample.py")], "0. 生成合成樣本（離線測試）")
    run([str(SRC / "build_history.py"), "--mode", "update" if a.latest else "build"],
        "2) 整合成主表")

    run([str(BASE / "analyze" / "frequency.py")], "3) 統計分析")

    if not a.no_model:
        run([str(BASE / "model" / "predict.py"), "--lookback", "10"], "4) 建模")


if __name__ == "__main__":
    main()
