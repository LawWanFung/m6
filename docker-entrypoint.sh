#!/bin/sh
# Dokploy / docker-compose 啟動時先跑呢個 entrypoint：
# 若數據卷（volume）內冇歷史主表，就從 image 內附帶嘅 seed 種子落去。
# 確保 fresh deploy / 新 volume 都有真實歷史數據，唔使開機就去抓數據。
set -e

echo "[entrypoint] start"
echo "[entrypoint] user: $(whoami) (uid=$(id -u))"
echo "[entrypoint] DATA_DIR=$DATA_DIR"

DATA_DIR="${DATA_DIR:-/app/data}"
SEED_DIR="${SEED_DIR:-/app/seeds}"
CSV="${CSV:-mark6_history.csv}"

echo "[entrypoint] mkdir -p $DATA_DIR"
mkdir -p "$DATA_DIR"

# ⚠️ 不再 chmod 777（舊版）。以 image 建 volume 時 ownership 會由 image 嘅
# /app/data（app:app）帶入；但若 volume 係舊 root 版 image 建嘅，就唔會可寫。
# 呢度只做「可寫性檢查 + 清晰診斷」：
if [ ! -w "$DATA_DIR" ]; then
    echo "[entrypoint] ERROR: $DATA_DIR 唔可以寫（user=$(id -u)）。" >&2
    echo "[entrypoint] 通常係舊版 image（root）建嘅 volume 仲喺度。修正方法：" >&2
    echo "[entrypoint]   1) 刪除舊 volume 重新部署（會先備份！）：" >&2
    echo "[entrypoint]        docker cp <container>:/app/data ./backup && docker volume rm mark6_app_data" >&2
    echo "[entrypoint]   2) 或者臨時以 root 跑一次改權：docker run --rm -v mark6_app_data:/d -u root alpine chown -R 65534:65534 /d" >&2
    exit 1
fi

# 只喺 volume 內冇主表時先複製（避免覆蓋已有的最新數據）
if [ ! -f "$DATA_DIR/$CSV" ] && [ -f "$SEED_DIR/$CSV" ]; then
    echo "[entrypoint] seeding..."
    cp "$SEED_DIR/$CSV" "$DATA_DIR/$CSV"
    echo "[entrypoint] seed done"
else
    echo "[entrypoint] skip seed"
fi

# 執行真正的啟動指令（Dockerfile CMD 或 docker-compose command）
echo "[entrypoint] exec $@"
exec "$@"
