#!/usr/bin/env sh
# Fly.io / Render set PORT; bind 0.0.0.0:$PORT (Fly matches fly.toml internal_port, e.g. 8080).
set -e
PORT="${PORT:-8080}"
export PORT
echo "[start] PORT=${PORT} binding 0.0.0.0:${PORT}"
exec gunicorn app:app \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 2 \
  --timeout 300 \
  --graceful-timeout 120 \
  --access-logfile - \
  --error-logfile -
