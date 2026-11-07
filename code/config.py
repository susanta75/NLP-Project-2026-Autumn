"""Paths and settings shared by all tasks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # code/
DATA_DIR = ROOT.parent / "data"
MODELS_DIR = ROOT.parent / "models"
TASK1_RESULTS = ROOT / "task1" / "results"
TASK2_RESULTS = ROOT / "task2" / "results"
TASK3_RESULTS = ROOT / "task3" / "results"
for d in (DATA_DIR, TASK1_RESULTS, TASK2_RESULTS, TASK3_RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# HaluEval QA at a fixed commit
HALUEVAL_URL = ("https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
                "b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json")
RAW_PATH = DATA_DIR / "qa_data.json"
ITEMS_PATH = DATA_DIR / "items.jsonl"  # two answers per sampled question
SAMPLES_PATH = TASK2_RESULTS / "resamples.jsonl"

SEED = 42
N_PAIRS = 200
N_RESAMPLES = 20
TEST_SIZE = 0.2

# Pretrained models, downloaded by download_models.py
LLM_PATH = MODELS_DIR / "Qwen2.5-3B-Instruct"                # Qwen/Qwen2.5-3B-Instruct
SYSTEM_PROMPT = "Answer the question concisely and factually."
MAX_NEW_TOKENS = 48
TEMPERATURE = 1.0

BERTSCORE_PATH = MODELS_DIR / "roberta-large"                # FacebookAI/roberta-large
BERTSCORE_LAYER = 17
BERTSCORE_BASELINE = MODELS_DIR / "bert_score_baseline" / "roberta-large.tsv"  # from bert-score 0.3.13
# the brief's microsoft/deberta-v3-large-mnli does not exist; this is the one selfcheckgpt uses
NLI_PATH = MODELS_DIR / "deberta-v3-large-mnli"              # potsawee/deberta-v3-large-mnli
SPACY_PATH = MODELS_DIR / "en_core_web_sm"                   # spaCy en_core_web_sm 3.8.0


def build_messages(question: str) -> list[dict]:
    """The model sees only the question, not HaluEval's knowledge text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def log_to(path):
    """Also write everything printed from now on to `path`."""
    sys.stdout = _Tee(sys.__stdout__, open(path, "w"))


def rel(path):
    """Path relative to the project root, for log messages."""
    return Path(path).resolve().relative_to(ROOT.parent)
