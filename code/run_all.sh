#!/usr/bin/env bash
# Runs Task 1, 2 and 3 in order. Each task saves its outputs in taskN/results/.
# Task 1 and 2 reuse their saved GPU outputs; pass --recompute to redo them:  bash run_all.sh --recompute
set -euo pipefail
cd "$(dirname "$0")"

# python: $PY if set, else the project's venv/, else python3
if [ -z "${PY:-}" ]; then
    if [ -x ../venv/bin/python ]; then PY=../venv/bin/python; else PY=python3; fi
fi

"$PY" prepare_data.py
"$PY" task1/task1.py "$@"
"$PY" task2/task2.py "$@"
"$PY" task3/task3.py
