"""Download HaluEval QA, draw a fixed random sample of pairs, and expand each pair
into two scoring instances: right_answer (label 0) and hallucinated_answer (label 1)."""
import json
import random
import urllib.request

from config import HALUEVAL_URL, ITEMS_PATH, N_PAIRS, RAW_PATH, SEED, rel


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


def ensure_items():
    """Called by every task, so each one also runs on a fresh clone."""
    if not ITEMS_PATH.exists():
        main()


if __name__ == "__main__":
    main()
