#!/bin/sh
# Container entrypoint: exec uvicorn so it becomes PID 1 and receives signals.
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn main:app --host "$HOST" --port "$PORT"
