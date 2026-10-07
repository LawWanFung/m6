"""pytest 配置：讓 `import model/... src/... analyze/...` 在倉庫根下可用。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def make_csv(tmp_path):
    """寫一個主表格式的 CSV，返回路徑。

    rows: list[list] — 每行 [draw_id, date, numbers("1,2,3,4,5,6"), extra_ball]
    """

    def _make(rows, name: str = "data.csv") -> Path:
        p = tmp_path / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["draw_id", "date", "numbers", "extra_ball"])
            w.writerows(rows)
        return p

    return _make
