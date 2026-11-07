#!/usr/bin/env bash
# Usage: bash run_all.sh [--recompute]
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PY:-}" ]; then
    if [ -x ../venv/bin/python ]; then PY=../venv/bin/python; else PY=python3; fi
fi

"$PY" prepare_data.py
"$PY" task1/task1.py "$@"
"$PY" task2/task2.py "$@"
"$PY" task3/task3.py
