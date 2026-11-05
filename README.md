# Project 7 — Do LLMs Know When They're Making Things Up? (HaluEval QA)

**Author:** Susanta Khamrui (26CS91R12) · Project 7, Group 21  
**Assigned dataset:** HaluEval, QA subset

Three hallucination detectors on 200 HaluEval QA questions. Each question gives two scored
items, its right answer (label 0) and its hallucinated answer (label 1), so there are 400 items.

| Task | Detector | Code |
|---|---|---|
| 1 — Baseline (30) | mean token log-prob and mean predictive entropy from forced decoding | [`code/task1/task1.py`](code/task1/task1.py) |
| 2 — Reproduction (30) | SelfCheckGPT: BERTScore and NLI against N = 20 resamples at T = 1.0 | [`code/task2/task2.py`](code/task2/task2.py) |
| 3 — Combine & improve (40) | logistic regression on the Task 1 + Task 2 features, 80/20 split | [`code/task3/task3.py`](code/task3/task3.py) |

The report is in [`report/report.pdf`](report/report.pdf) (source: `report/report.tex`).

## Results at a glance

AUC-PR (random = 0.50, the classes are balanced).

| Detector | All 400 items | Held-out 20% (80 items) |
|---|---|---|
| Task 1: −mean log-prob | 0.406 | 0.429 |
| Task 1: mean entropy | 0.417 | 0.459 |
| Task 1: LR(log-prob, entropy) | — | 0.602 |
| Task 2: SelfCheck-BERTScore | 0.317 | 0.323 |
| Task 2: SelfCheck-NLI | 0.653 | 0.651 |
| Task 2: LR(BERTScore, NLI) | — | 0.957 |
| **Task 3: LR ensemble (4 features)** | — | **0.951** |
| Reference: answer length alone | 0.972 | 0.965 |

Hallucinated answers in HaluEval are about four times longer than right answers, so length
alone is almost a perfect detector. Task 3 also measures how much signal is left once length
is controlled for (about 0.64 AUC-PR).

## Quick start (Docker, one command)

From the project root, with Docker installed:

```bash
bash run_docker.sh               # builds the image, then runs Task 1, 2 and 3 (CPU, about 15 s after the build)
bash run_docker.sh --recompute   # also redoes the Task 1 and Task 2 GPU steps
```

- The first build downloads the pinned Python packages (the image is about 10 GB, mostly the
  CUDA build of PyTorch) and the HaluEval data, so it takes a few minutes. Later runs reuse it.
- `code/` is mounted into the container, so all outputs are written to `code/taskN/results/`
  on the host. The run produces the same CSV and log files as running without Docker.
- `--recompute` needs an NVIDIA GPU and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  The model checkpoints are cached in `~/.cache/huggingface` on the host, so they are
  downloaded only once.

**GPU inside Docker.** Without the toolkit, `--recompute` stops with a message (Docker's own
error is `failed to discover GPU vendor from CDI`). On Ubuntu, install it once:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi   # should list the GPU
```

The GPU steps can also be rerun without Docker: `bash code/run_all.sh --recompute`.

To run without Docker, see [Environment setup](#environment-setup) and [Running](#running).

## Repository layout

There is one script per task. Each writes all its outputs (CSVs, figures, and a
`taskN_results.txt` with everything it printed) to its own `results/` folder.

```
README.md            this file
requirements.txt     pinned pip environment
Dockerfile           the same environment as a Docker image (python:3.12.3-slim + requirements.txt)
run_docker.sh        builds the image and runs all tasks in it
report/              report.tex and report.pdf
code/
  config.py          every path, seed, model name and hyperparameter
  prepare_data.py    downloads HaluEval QA and draws the 200-question sample
  run_all.sh         runs Task 1, 2 and 3 in order
  task1/task1.py     forced decoding -> analysis -> figures            outputs: task1/results/
  task2/task2.py     resampling -> SelfCheckGPT -> analysis -> figures outputs: task2/results/
  task3/task3.py     ensemble -> length control -> figures             outputs: task3/results/
data/                created by prepare_data.py (not committed)
```

## Environment setup

Tested on Ubuntu 24.04 with Python 3.12.3 and an NVIDIA RTX 5060 Ti (16 GB, driver 595.99, CUDA 13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is needed only to recompute the Task 1 and Task 2 model outputs. The 3B model in
float16 needs about 7 GB of VRAM, and nothing is trained or fine-tuned. Those outputs are
committed, so without `--recompute` every script runs on the CPU.

## Data

Only the assigned dataset is used. The project brief gives each group exactly one of WikiBio,
TruthfulQA and HaluEval, so the other two are not downloaded.

Every task script prepares the data itself if `data/items.jsonl` is missing. To do it by hand:

```bash
cd code
python prepare_data.py
```

This:
1. Downloads `qa_data.json` (10,000 QA pairs, ~6 MB) from the official HaluEval repository
   (<https://github.com/RUCAIBox/HaluEval>, file `data/qa_data.json`) into `data/`.
2. Samples 200 pairs with `random.Random(42)`.
3. Writes the 400 scored items to `data/items.jsonl` as `{pair_id, question, answer, label}`.

Prompts are closed-book: the model sees only the question, not HaluEval's `knowledge` field.

## Running

All commands are run from `code/`.

```bash
bash run_all.sh                  # all three tasks in order, reusing saved model outputs
bash run_all.sh --recompute      # redo the GPU steps of Task 1 and Task 2 from scratch
```

`run_all.sh` uses `../.venv/bin/python` if it exists, otherwise `python3`. Set `PY=...` to use
a different interpreter.

Each task can also be run on its own (with the virtual environment active):

```bash
python task1/task1.py [--recompute]   # forced decoding (GPU) -> AUC-PR, figures
python task2/task2.py [--recompute]   # 20 resamples + SelfCheckGPT (GPU) -> AUC-PR, figures
python task3/task3.py                 # CPU only; needs the Task 1 and Task 2 features
```

- **Without `--recompute`**, Task 1 and Task 2 load their saved features and resamples, then
  redo only the analysis and figures. This takes seconds.
- **With `--recompute`**, Task 1 reruns forced decoding (about 10 s on the GPU above), and
  Task 2 regenerates all 4,000 samples and rescores them. If resampling is interrupted, it
  resumes on the next run.
- **Task 3** always recomputes everything. It stops with a clear message if the Task 1 or
  Task 2 features are missing.

**Reproducibility check.** Running both GPU tasks with `--recompute` reproduced the committed
outputs exactly. The Task 1 features match with a difference of 0. For Task 2, all 200
questions got identical samples and both SelfCheck scores match with a difference of 0.

| Script | Outputs in `taskN/results/` |
|---|---|
| `task1.py` | `task1_features.csv`, `task1_results.txt`, `pr_curves.png`, `roc_curves.png`, `score_distributions.png`, `length_confound.png` |
| `task2.py` | `resamples.jsonl`, `task2_features.csv`, `task2_results.txt`, `pr_curves.png`, `roc_curves.png`, `score_distributions.png`, `auc_by_knowledge.png` |
| `task3.py` | `task3_results.csv`, `test_predictions.csv`, `task3_coefficients.json`, `task3_length_control.csv`, `task3_repeated_splits.csv`, `task3_results.txt`, `pr_curves.png`, `roc_curves.png`, `auc_comparison.png`, `length_control.png`, `coefficients.png` |

To rebuild the report after changing results or text:

```bash
cd report && latexmk -pdf report.tex
```

The report reads its figures directly from `code/taskN/results/`.

## Random seeds

All seeds are fixed in [`code/config.py`](code/config.py) (`SEED = 42`) or set where they are used.

| What | Seed |
|---|---|
| Sampling the 200 HaluEval pairs | `random.Random(42)` |
| Task 2 resampling (Qwen generation) | `torch.manual_seed(42 + pair_id)` before each question |
| Task 3 80/20 split (by question) | `train_test_split(random_state=42)` |
| Logistic regression | `LogisticRegression(random_state=42)` (lbfgs is deterministic anyway) |
| Bootstrap CIs (1000 resamples of test questions) | `np.random.default_rng(42)` |
| Task 3 length check, 100 extra splits | `random_state = 0 … 99` |
| Jitter in strip plots (cosmetic only) | `np.random.default_rng(0)` |

Task 1 forced decoding and the SelfCheck scoring involve no randomness.

## Pretrained checkpoints

All checkpoints are downloaded automatically from the Hugging Face Hub on first use.

| Used for | Checkpoint | Source |
|---|---|---|
| Task 1 forced decoding, Task 2 resampling | `Qwen/Qwen2.5-3B-Instruct` (float16) | <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct> |
| SelfCheck-BERTScore backbone | `roberta-large`, layer 17, baseline rescaling | <https://huggingface.co/FacebookAI/roberta-large> (via `bert-score` 0.3.13) |
| SelfCheck-NLI | `potsawee/deberta-v3-large-mnli` | <https://huggingface.co/potsawee/deberta-v3-large-mnli> |
| Sentence splitting inside SelfCheckGPT | spaCy `en_core_web_sm` 3.8.0 | <https://github.com/explosion/spacy-models> |
| SelfCheckGPT code | `selfcheckgpt` 0.1.7 | <https://github.com/potsawee/selfcheckgpt> (commit `19b492a`) |

## Deviations from the specification

- **NLI checkpoint.** The brief names `microsoft/deberta-v3-large-mnli`, which does not exist
  on the Hugging Face Hub. `potsawee/deberta-v3-large-mnli` is used instead. It is the
  DeBERTa-v3-large MNLI checkpoint that the official SelfCheckGPT code loads.
- **LLM size.** Qwen2.5-3B-Instruct, so that everything runs on a single 16 GB GPU.
- **Sample size.** 200 of the 10,000 HaluEval QA pairs (400 items), because of the compute
  needed for 20 resamples per question.
- **Compatibility shim.** `patch_for_transformers5()` in `code/task2/task2.py` restores
  `batch_encode_plus`, which transformers 5 removed but `selfcheckgpt` 0.1.7 still calls. It
  also caches the BERTScore model between calls. With the shim, the official code reproduces
  the example outputs in its README to 6 decimal places.
- **Unscorable items.** SelfCheck-BERTScore cannot score 2 questions (4 items) whose 20 samples
  are all 3 tokens or shorter. They are left out of the Task 2 BERTScore AUC-PR (n = 396) and
  filled with the training-set median in Task 3.
