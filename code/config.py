"""Shared settings for Project 7 (HaluEval QA) — every script imports from here."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent             # code/
DATA_DIR = ROOT.parent / "data"                     # downloaded data stays outside code/ (not committed)
TASK1_RESULTS = ROOT / "task1" / "results"
TASK2_RESULTS = ROOT / "task2" / "results"
TASK3_RESULTS = ROOT / "task3" / "results"
for d in (DATA_DIR, TASK1_RESULTS, TASK2_RESULTS, TASK3_RESULTS):
    d.mkdir(parents=True, exist_ok=True)

HALUEVAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
RAW_PATH = DATA_DIR / "qa_data.json"
ITEMS_PATH = DATA_DIR / "items.jsonl"               # 2 scoring instances per sampled pair
SAMPLES_PATH = TASK2_RESULTS / "resamples.jsonl"    # N stochastic answers per question

SEED = 42            # used for the pair sample, the resampling RNG and the 80/20 split
N_PAIRS = 200        # 200 pairs -> 400 scored items
N_RESAMPLES = 20
TEST_SIZE = 0.2

# Open LLM used for both forced decoding (Task 1) and resampling (Task 2).
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Answer the question concisely and factually."
MAX_NEW_TOKENS = 48
TEMPERATURE = 1.0

BERTSCORE_MODEL = "roberta-large"
# The spec names microsoft/deberta-v3-large-mnli, which does not exist on the HF Hub.
# This is the DeBERTa-v3-large MNLI checkpoint used by the official SelfCheckGPT code
# (binary head: index 0 = entailment, index 1 = contradiction).
NLI_MODEL = "potsawee/deberta-v3-large-mnli"


def build_messages(question: str) -> list[dict]:
    """Closed-book prompt: the model sees only the question, not HaluEval's `knowledge`."""
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

    def __getattr__(self, name):  # isatty(), encoding, ... come from the real terminal
        return getattr(self.streams[0], name)


def log_to(path):
    """From here on, everything printed also goes into `path` (overwritten on every run)."""
    sys.stdout = _Tee(sys.__stdout__, open(path, "w"))


def rel(path):
    """Path relative to the project root, so logs read the same on any machine or in Docker."""
    return Path(path).resolve().relative_to(ROOT.parent)
