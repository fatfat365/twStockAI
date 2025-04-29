#!/usr/bin/env sh
# 如果沒有 PORT 環境變數，就給個預設 8000
: "${PORT:=5555}"

exec uvicorn twStockSrvXGBoots:app \
  --host 0.0.0.0 \
  --port "$PORT"
