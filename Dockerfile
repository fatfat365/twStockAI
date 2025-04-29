# 使用官方 Python slim 映像
FROM python:3.10-slim

# 1) 安裝編譯工具與必要套件
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2) 下載並編譯 TA-Lib C 原生庫
RUN wget -qO /tmp/ta-lib-0.4.0-src.tar.gz \
      http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    mkdir -p /usr/src/ta-lib && \
    tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /usr/src/ta-lib --strip-components=1 && \
    cd /usr/src/ta-lib && \
      ./configure --prefix=/usr && make && make install && \
    rm -rf /tmp/ta-lib-0.4.0-src.tar.gz /usr/src/ta-lib

# 3) 設定工作目錄並複製需求檔
WORKDIR /app
COPY requirements.txt .

# 4) 安裝 Python 套件（包含 ta-lib binding）
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5) 複製程式碼
COPY . /app

# 6) 啟動指令 (依專案需求調整)
#   這裡以 Gunicorn 啟動 FastAPI 為例
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:$PORT", "--workers", "4"]
