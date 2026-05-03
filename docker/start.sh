#!/usr/bin/env sh
# Render injects PORT; must listen on 0.0.0.0:$PORT — see https://render.com/docs/web-services#port-binding
set -e
PORT="${PORT:-5000}"
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
