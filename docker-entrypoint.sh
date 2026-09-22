#!/bin/sh
# Dokploy / docker-compose 啟動時先跑呢個 entrypoint：
# 若數據卷（volume）內冇歷史主表，就從 image 內附帶嘅 seed 種子落去。
# 確保 fresh deploy / 新 volume 都有真實歷史數據，唔使開機就去抓數據。
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
SEED_DIR="${SEED_DIR:-/app/seeds}"
CSV="${CSV:-mark6_history.csv}"

mkdir -p "$DATA_DIR"

# 只喺 volume 內冇主表時先複製（避免覆蓋已有的最新數據）
if [ ! -f "$DATA_DIR/$CSV" ] && [ -f "$SEED_DIR/$CSV" ]; then
    echo "[entrypoint] Seeding $DATA_DIR/$CSV from image seed..."
    cp "$SEED_DIR/$CSV" "$DATA_DIR/$CSV"
fi

# 執行真正的啟動指令（Dockerfile CMD 或 docker-compose command）
exec "$@"
