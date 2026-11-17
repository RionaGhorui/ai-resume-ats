#!/usr/bin/env bash
# Starts the API (internal) and the UI (public). If either one dies, the container exits,
# so the platform (Cloud Run, Docker restart policy, ...) restarts it instead of serving a broken app.
set -euo pipefail

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

streamlit run frontend/app.py \
  --server.port "${PORT:-8080}" --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null || true' TERM INT   # forward Ctrl+C / platform shutdown
wait -n                                                         # returns when the first process exits
status=$?
kill "$API_PID" "$UI_PID" 2>/dev/null || true
exit "$status"
