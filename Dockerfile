FROM python:3.14-slim

# ⚠️ base image 用 3.14：同 requirements.txt 註釋一樣，本地/測試/容器統一
# 解釋器版本。舊版 3.12 + 未鎖定依賴 → 容器裝到同本地唔同嘅版本組合。

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# ⚠️ 以非特權用戶運行（舊版用 root + entrypoint chmod 777 data/）。
# 容器逃逸時攻擊者拿到嘅係普通用戶，唔係 root。
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

# 先裝依賴（利用 Docker 快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼（owner = app 用戶）
COPY --chown=app:app . .

# 開放端口（Dokploy 會將流量轉發到此）
EXPOSE 8000

# 將歷史數據複製到 volume 以外嘅位置，供 entrypoint 開機種子用
COPY data/mark6_history.csv /app/seeds/mark6_history.csv

# 啟動前先種子：把 image 內附帶嘅歷史數據寫入數據卷（volume）
# 確保 fresh deploy / 新 volume 都有真實歷史數據，唔使開機就去抓
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R app:app /app

USER app
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Dokploy 以 Docker 方式部署時會執行此 CMD。
# 使用 $PORT（Dokploy 會設定），預設 8000。
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
