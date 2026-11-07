# Project 7: Do LLMs Know When They're Making Things Up? (HaluEval QA)

**Author:** Susanta Khamrui (26CS91R12) · Project 7, Group 21  
**Dataset:** HaluEval, QA subset

The project tests three ways to detect hallucinated answers, on 200 HaluEval questions. Each
question has a right answer (label 0) and a hallucinated answer (label 1), so 400 answers are
scored.

| Task | Method | Code |
|---|---|---|
| 1 | Token log-probability and entropy of each answer under the LLM | [`code/task1/task1.py`](code/task1/task1.py) |
| 2 | SelfCheckGPT: compare each answer with 20 sampled answers (BERTScore and NLI) | [`code/task2/task2.py`](code/task2/task2.py) |
| 3 | Logistic regression on the Task 1 and Task 2 scores (80/20 split) | [`code/task3/task3.py`](code/task3/task3.py) |

The report is [`report/report.pdf`](report/report.pdf).

## Results

AUC-PR on the held-out 20% (80 answers). Random guessing gives 0.50.

| Detector | AUC-PR |
|---|---|
| Task 1 (log-prob + entropy) | 0.602 |
| Task 2 (BERTScore + NLI) | 0.957 |
| Task 3 (all four scores) | 0.951 |
| Answer length alone | 0.965 |

Hallucinated answers are much longer than right ones, so length alone works almost perfectly.
With length removed, the Task 1 and Task 2 scores still reach about 0.64.

## Quick start (Docker)

```bash
bash run_docker.sh               # runs Task 1, 2 and 3
bash run_docker.sh --recompute   # also reruns the GPU steps of Task 1 and Task 2
```

The first run builds the image (about 10 GB, a few minutes). Results are written to
`code/taskN/results/` on your machine. `--recompute` needs an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Setup without Docker

Tested on Ubuntu 24.04, Python 3.12.3, NVIDIA RTX 5060 Ti (16 GB).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins every package (107) to the version used for the results.

A GPU is needed only with `--recompute`. Without it, the scripts reuse the saved model outputs
and run on the CPU in seconds.

## Datasets

The brief assigns each group one dataset, so every task uses only HaluEval QA. The command
below can also download the other two Project 7 datasets, but no task uses them.

| Dataset | Used | Source | Command (in `code/`) |
|---|---|---|---|
| HaluEval QA | yes | [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval), `data/qa_data.json` | `python prepare_data.py` |
| WikiBio GPT-3 | no | [potsawee/wiki_bio_gpt3_hallucination](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination) | `python prepare_data.py --all-datasets` |
| TruthfulQA (generation) | no | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | `python prepare_data.py --all-datasets` |

`prepare_data.py` downloads `qa_data.json` (10,000 questions) into `data/`, picks 200 questions
with seed 42, and writes the 400 answers to `data/items.jsonl`. It runs automatically when
needed. The model sees only the question, not HaluEval's `knowledge` text.

## Pretrained models

All models are downloaded into `models/` and loaded from there. Task 1 and Task 2 download
them automatically when run with `--recompute`. To download them by hand (about 9 GB):
`cd code && python download_models.py`.

| Used for | Model | Source |
|---|---|---|
| Task 1 scoring, Task 2 sampling | Qwen2.5-3B-Instruct (float16) | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| SelfCheck-BERTScore | roberta-large (layer 17) | [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) |
| BERTScore rescaling file | roberta-large.tsv | from the [bert-score](https://github.com/Tiiiger/bert_score) 0.3.13 package |
| SelfCheck-NLI | DeBERTa-v3-large MNLI | [potsawee/deberta-v3-large-mnli](https://huggingface.co/potsawee/deberta-v3-large-mnli) |
| Sentence splitting | spaCy en_core_web_sm 3.8.0 | [explosion/spacy-models](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) |

The exact versions are pinned in `code/download_models.py`. SelfCheckGPT itself is the
official code ([potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt)), installed
with pip as `selfcheckgpt==0.1.7`. No model is trained or fine-tuned.

## Running

Run from `code/`:

```bash
bash run_all.sh                  # all tasks in order: data -> Task 1 -> Task 2 -> Task 3
bash run_all.sh --recompute      # same, but redo the GPU steps of Task 1 and Task 2

python task1/task1.py [--recompute]   # Task 1 only
python task2/task2.py [--recompute]   # Task 2 only
python task3/task3.py                 # Task 3 only (uses the saved Task 1 and Task 2 scores)
```

- Without `--recompute`, Task 1 and Task 2 load their saved scores and only redo the
  analysis and figures.
- With `--recompute`, Task 1 takes about 10 s and Task 2 about 7 minutes on the GPU above.
- Each task can run on its own on a fresh copy of the repository.

Rerunning everything with `--recompute` gives exactly the same outputs as the saved ones.

## Outputs

| Folder | Contents |
|---|---|
| `code/task1/results/` | Task 1 scores (`task1_features.csv`), printed results (`task1_results.txt`), figures |
| `code/task2/results/` | the 4,000 sampled answers (`resamples.jsonl`), Task 2 scores, printed results, figures |
| `code/task3/results/` | AUC-PR table with confidence intervals, test predictions, regression weights, length checks, figures |
| `data/` | downloaded and prepared data |
| `models/` | downloaded pretrained models |

The only trained model is the Task 3 logistic regression. It is refitted on every run, and its
weights are saved to `code/task3/results/task3_coefficients.json`.

To rebuild the report: `cd report && latexmk -pdf report.tex`.

## Random seeds

| Step | Seed |
|---|---|
| Choosing the 200 questions | `random.Random(42)` |
| Task 2 sampling | `torch.manual_seed(42 + question id)` |
| Task 3 80/20 split and logistic regression | `random_state=42` |
| Bootstrap confidence intervals | `numpy.random.default_rng(42)` |
| 100 extra splits (length check) | `random_state` 0 to 99 |

## Changes from the brief

- **NLI model:** the brief names `microsoft/deberta-v3-large-mnli`, which does not exist on
  Hugging Face. `potsawee/deberta-v3-large-mnli` is used instead; it is the model the official
  SelfCheckGPT code loads.
- **LLM size:** a 3B model, so that everything fits on one 16 GB GPU.
- **Sample size:** 200 of the 10,000 questions, because Task 2 needs 20 samples per question.
- **Small code fix:** `selfcheckgpt` 0.1.7 calls a function removed in transformers 5.
  `patch_official_code()` in `task2.py` adds it back and loads the models from `models/`.
  It does not change any score.
- **Missing scores:** BERTScore cannot score 2 questions (4 answers) whose samples are all very
  short. They are left out of the Task 2 BERTScore result and filled with the training median
  in Task 3.
