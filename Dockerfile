FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# 先裝依賴（利用 Docker 快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 開放端口（Dokploy 會將流量轉發到此）
EXPOSE 8000

# 將歷史數據複製到 volume 以外嘅位置，供 entrypoint 開機種子用
COPY data/mark6_history.csv /app/seeds/mark6_history.csv

# 啟動前先種子：把 image 內附帶嘅歷史數據寫入數據卷（volume）
# 確保 fresh deploy / 新 volume 都有真實歷史數據，唔使開機就去抓
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Dokploy 以 Docker 方式部署時會執行此 CMD。
# 使用 $PORT（Dokploy 會設定），預設 8000。
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
