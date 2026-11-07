#!/usr/bin/env bash
# Usage: bash run_docker.sh [--recompute]   (--recompute needs the NVIDIA Container Toolkit)
set -euo pipefail
cd "$(dirname "$0")"

gpu=()
if [[ " $* " == *" --recompute "* ]]; then
    gpu=(--gpus all)
    if ! command -v nvidia-ctk >/dev/null; then
        echo "--recompute in Docker needs the NVIDIA Container Toolkit." >&2
        echo "Install it, or run without Docker:" >&2
        echo "  bash code/run_all.sh --recompute" >&2
        exit 1
    fi
fi

docker build -t halueval-p7 .

# mount code/ (outputs) and models/ (pretrained models)
mkdir -p models
docker run --rm ${gpu[@]+"${gpu[@]}"} \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/code:/app/code" \
    -v "$PWD/models:/app/models" \
    halueval-p7 "$@"
