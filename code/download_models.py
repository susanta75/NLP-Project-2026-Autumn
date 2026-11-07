"""Download the pretrained models into models/ (pinned versions).

Usage: python download_models.py [llm|bertscore|nli|spacy ...]   (default: all)
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import BERTSCORE_BASELINE, BERTSCORE_PATH, LLM_PATH, NLI_PATH, SPACY_PATH, rel

# local folder -> [(Hugging Face repo, commit, files)]
HF_FILES = {
    LLM_PATH: [("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                ["config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "tokenizer_config.json", "tokenizer.json", "vocab.json"])],
    BERTSCORE_PATH: [("FacebookAI/roberta-large", "722cf37b1afa9454edce342e7895e588b6ff1d59",
                      ["config.json", "merges.txt", "model.safetensors", "tokenizer_config.json",
                       "tokenizer.json", "vocab.json"])],
    # model.safetensors is only on the PR #2 branch of this repo
    NLI_PATH: [("potsawee/deberta-v3-large-mnli", "83f4359862aba52c07506cd23d4454db77945b36",
                ["added_tokens.json", "config.json", "special_tokens_map.json", "spm.model",
                 "tokenizer_config.json"]),
               ("potsawee/deberta-v3-large-mnli", "924531a98c784ffc1eb18d0765f0e2b2af7c8866",
                ["model.safetensors"])],
}
SPACY_WHEEL = ("https://github.com/explosion/spacy-models/releases/download/"
               "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")


def fetch_hf(folder):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download
    for repo, revision, files in HF_FILES[folder]:
        for name in files:
            if not (folder / name).exists():
                hf_hub_download(repo, name, revision=revision, local_dir=folder)
    shutil.rmtree(folder / ".cache", ignore_errors=True)


def fetch_spacy():
    if (SPACY_PATH / "meta.json").exists():
        return
    print(f"Downloading {SPACY_WHEEL}")
    prefix = "en_core_web_sm/en_core_web_sm-3.8.0/"
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(SPACY_WHEEL).read())) as wheel:
        for member in wheel.namelist():
            if member.startswith(prefix) and not member.endswith("/"):
                target = SPACY_PATH / member[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(member))


def fetch_bertscore_baseline():
    if BERTSCORE_BASELINE.exists():
        return
    import bert_score  # the baseline file ships with the package
    BERTSCORE_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(bert_score.__file__).parent / "rescale_baseline/en/roberta-large.tsv",
                BERTSCORE_BASELINE)


MODELS = {
    "llm": lambda: fetch_hf(LLM_PATH),
    "bertscore": lambda: (fetch_hf(BERTSCORE_PATH), fetch_bertscore_baseline()),
    "nli": lambda: fetch_hf(NLI_PATH),
    "spacy": fetch_spacy,
}


def ensure_models(*names):
    for name in names or MODELS:
        MODELS[name]()


if __name__ == "__main__":
    ensure_models(*sys.argv[1:])
    print(f"Models ready in {rel(LLM_PATH.parent)}")
