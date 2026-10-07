#!/bin/sh
# Dokploy / docker-compose 啟動時先跑呢個 entrypoint。
#
# 做四件事：
#   1. 確保 $DATA_DIR 同 $DATA_DIR/raw 存在
#   2. **自動修復 ownership**：volume 若係舊 root image 建嘅（root 擁有），
#      chown 返 app:app —— 否則 app 用戶寫唔入（實測：fetch 寫 last_all.json
#      時 PermissionError，2026-10-07 部署事故嘅根因）
#   3. 可寫性檢查（修唔到就 fail loudly，唔好令容器「表面健康」）
#   4. volume 冇主表時由 seed 種子；最後 gosu 降權 app 先 exec 應用
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
SEED_DIR="${SEED_DIR:-/app/seeds}"
CSV="${CSV:-mark6_history.csv}"

echo "[entrypoint] start (uid=$(id -u), user=$(whoami))"
echo "[entrypoint] DATA_DIR=$DATA_DIR"

mkdir -p "$DATA_DIR" "$DATA_DIR/raw"

# ── 2. ownership 自動修復（只係 root 身份先做得到） ─────────────────────────
APP_UID="$(id -u app 2>/dev/null || echo 0)"
if [ "$(id -u)" = "0" ]; then
    needs=0
    for p in "$DATA_DIR" "$DATA_DIR/raw" "$DATA_DIR/$CSV"; do
        if [ -e "$p" ]; then
            if [ "$(stat -c %u "$p" 2>/dev/null || echo 0)" != "$APP_UID" ]; then
                needs=1
            fi
        fi
    done
    if [ "$needs" = "1" ]; then
        echo "[entrypoint] 偵測到 volume 有非 app 擁有嘅檔案/目錄（通常係舊 root image 建嘅）"
        echo "[entrypoint] chown -R app:app $DATA_DIR（自動修復）"
        chown -R app:app "$DATA_DIR"
    fi
fi

# ── 3. 可寫性檢查（root 修完後都要再驗一次；只係非 root 跑時呢度攔截） ───
for d in "$DATA_DIR" "$DATA_DIR/raw"; do
    if ! touch "$d/.writetest" 2>/dev/null; then
        echo "[entrypoint] ERROR: $d 唔可以寫（uid=$(id -u)）。" >&2
        echo "[entrypoint] 通常係舊 root image 建嘅 volume。修正方法：" >&2
        echo "[entrypoint]   1) 刪除舊 volume 重新部署（會先備份！）：" >&2
        echo "[entrypoint]        docker cp <container>:/app/data ./backup && docker volume rm mark6_app_data" >&2
        echo "[entrypoint]   2) 或者以 root 手動改權一次：docker run --rm -v mark6_app_data:/d -u root alpine chown -R $APP_UID:$APP_UID /d" >&2
        exit 1
    fi
    rm -f "$d/.writetest"
done

# ── 4. 只喺 volume 內冇主表時先複製 seed（避免覆蓋已有嘅最新數據） ─────────
if [ ! -f "$DATA_DIR/$CSV" ] && [ -f "$SEED_DIR/$CSV" ]; then
    echo "[entrypoint] seeding..."
    cp "$SEED_DIR/$CSV" "$DATA_DIR/$CSV"
    if [ "$(id -u)" = "0" ]; then
        chown app:app "$DATA_DIR/$CSV"
    fi
    echo "[entrypoint] seed done"
else
    echo "[entrypoint] skip seed"
fi

# ── 降權：root → app（應用本身全程非 root） ──────────────────────────────
if [ "$(id -u)" = "0" ]; then
    echo "[entrypoint] exec (as app): $*"
    exec gosu app "$@"
else
    echo "[entrypoint] exec (as $(whoami)): $*"
    exec "$@"
fi
