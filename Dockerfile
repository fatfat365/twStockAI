# 1) 基底映像
FROM python:3.10-slim

# 2) 安裝編譯工具 & C 原生庫依賴
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

# 4) 設定工作目錄、複製 requirements
WORKDIR /app
COPY requirements.txt .

# 5) 升級 pip 工具並安裝 TA-Lib binding（無隔離編譯）
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-build-isolation TA-Lib==0.4.24

# 6) 安裝其餘套件
RUN pip install -r requirements.txt

# 7) 複製並啟動專案
COPY . /app
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:$PORT", "--workers", "4"]
