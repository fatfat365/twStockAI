#!/usr/bin/env sh
: "${PORT:=5555}"

exec uvicorn twStockSrvXGBoots:app \
  --host 0.0.0.0 \
  --port "$PORT"
