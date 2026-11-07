"""Prepare the data. Run: python prepare_data.py [--all-datasets]

HaluEval QA (the dataset assigned to this group, and the only one any task uses): download
qa_data.json, draw a fixed random sample of pairs, and expand each pair into two scoring
instances: right_answer (label 0) and hallucinated_answer (label 1).

--all-datasets also downloads the raw files of Project 7's other two datasets, WikiBio GPT-3 and
TruthfulQA, into data/ (pinned versions, sources as in the project's Datasets document). No task
reads them: each group works only on its one assigned dataset.
"""
import argparse
import json
import os
import random
import shutil
import urllib.request

from config import DATA_DIR, HALUEVAL_URL, ITEMS_PATH, N_PAIRS, RAW_PATH, SEED, rel

# local folder -> (Hugging Face dataset repo, pinned commit, file)
OTHER_DATASETS = {
    DATA_DIR / "wiki_bio_gpt3_hallucination": (
        "potsawee/wiki_bio_gpt3_hallucination", "b3cfb73209a8c51582fa1d9b7fe7e45fec5529b2",
        "data/evaluation-00000-of-00001-e91191b8ff41afbe.parquet"),
    DATA_DIR / "truthful_qa": (  # configuration "generation", as the Datasets document specifies
        "truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "generation/validation-00000-of-00001.parquet"),
}


def main():
    if not RAW_PATH.exists():
        print(f"Downloading {HALUEVAL_URL}")
        urllib.request.urlretrieve(HALUEVAL_URL, RAW_PATH)

    with open(RAW_PATH) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"HaluEval QA: {len(pairs)} pairs")

    idx = sorted(random.Random(SEED).sample(range(len(pairs)), N_PAIRS))
    with open(ITEMS_PATH, "w") as f:
        for pair_id in idx:
            p = pairs[pair_id]
            for answer, label in ((p["right_answer"], 0), (p["hallucinated_answer"], 1)):
                f.write(json.dumps({
                    "pair_id": pair_id,
                    "question": p["question"],
                    "answer": answer,
                    "label": label,
                }) + "\n")
    print(f"Wrote {2 * N_PAIRS} items ({N_PAIRS} pairs, seed={SEED}) to {rel(ITEMS_PATH)}")


def download_other_datasets():
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # plain download: no chunk cache left behind in ~/.cache
    from huggingface_hub import hf_hub_download
    for folder, (repo, revision, name) in OTHER_DATASETS.items():
        if not (folder / name).exists():
            hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=folder)
        shutil.rmtree(folder / ".cache", ignore_errors=True)  # download bookkeeping, not data
        print(f"{repo}: {rel(folder / name)}")


def ensure_items():
    """Called by every task, so each one also runs on a fresh clone."""
    if not ITEMS_PATH.exists():
        main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all-datasets", action="store_true",
                        help="also download the raw WikiBio GPT-3 and TruthfulQA files (not used by any task)")
    args = parser.parse_args()
    main()
    if args.all_datasets:
        download_other_datasets()
