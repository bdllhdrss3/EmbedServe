#!/usr/bin/env bash
set -e

exec pip install --no-cache-dir -r requirements.txt
exec uvicorn app:app --reload --host 0.0.0.0 --port 8000
