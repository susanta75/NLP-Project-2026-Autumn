#!/usr/bin/env bash
# One-command run inside Docker. Results appear in code/taskN/results/ on the host.
#   bash run_docker.sh               all three tasks from the saved model outputs (CPU, ~1 min)
#   bash run_docker.sh --recompute   also redo the Task 1 and Task 2 GPU steps
#                                    (needs an NVIDIA GPU and the NVIDIA Container Toolkit)
set -euo pipefail
cd "$(dirname "$0")"

gpu=()
if [[ " $* " == *" --recompute "* ]]; then
    gpu=(--gpus all)
    # Docker needs the NVIDIA Container Toolkit to pass the GPU into the container.
    if ! command -v nvidia-ctk >/dev/null; then
        echo "--recompute in Docker needs the NVIDIA Container Toolkit, which is not installed." >&2
        echo "Install it (see README, 'GPU inside Docker'), or run without Docker:" >&2
        echo "  bash code/run_all.sh --recompute" >&2
        exit 1
    fi
fi

docker build -t halueval-p7 .

# The code folder is mounted so outputs land on the host; the Hugging Face cache is
# mounted so the model checkpoints are downloaded only once.
mkdir -p "$HOME/.cache/huggingface"
docker run --rm ${gpu[@]+"${gpu[@]}"} \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/code:/app/code" \
    -v "$HOME/.cache/huggingface:/hf-cache" \
    halueval-p7 "$@"
