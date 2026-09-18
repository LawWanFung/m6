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

# Dokploy 以 Docker 方式部署時會執行此 CMD。
# 使用 $PORT（Dokploy 會設定），預設 8000。
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
