# 1) 基底映像
FROM python:3.10-slim

# 2) 安裝系統編譯工具 & CA 憑證
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 3) 下載並編譯 TA-Lib C 原生庫
RUN wget -qO /tmp/ta-lib-0.4.0-src.tar.gz \
      http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    mkdir -p /usr/src/ta-lib && \
    tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /usr/src/ta-lib --strip-components=1 && \
    cd /usr/src/ta-lib && \
      ./configure --prefix=/usr && \
      make && \
      make install && \
    rm -rf /tmp/ta-lib-0.4.0-src.tar.gz /usr/src/ta-lib

# 4) 複製 requirements
WORKDIR /app
COPY requirements.txt .

# 5) 升級 pip/setuptools/​wheel，先安裝指定版本的 NumPy
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.23.5

# 6) 單獨安裝 TA-Lib binding（使用本機 C lib、無 build 隔離）
RUN pip install --no-build-isolation TA-Lib==0.4.24

# 7) 安裝其他相依套件
RUN pip install -r requirements.txt

COPY start.sh /app/start.sh
COPY . /app
WORKDIR /app

# 加上執行權限
RUN chmod +x start.sh

# shell form → 透過 sh 啟動，$PORT 才會被展開
ENTRYPOINT /app/start.sh
