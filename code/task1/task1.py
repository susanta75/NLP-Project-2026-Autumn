"""Task 1: token log-probability and entropy of each answer under the LLM.

Usage: python task1/task1.py [--recompute]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import precision_recall_curve, roc_auc_score

from config import ITEMS_PATH, LLM_PATH, TASK1_RESULTS, build_messages, log_to, rel
from download_models import ensure_models
from prepare_data import ensure_items

OUT = TASK1_RESULTS
FEATURES = OUT / "task1_features.csv"
# column, sign (higher = more suspicious), label, colour, line style
SCORES = [("mean_logprob", -1, "−mean log-prob (spec)", "#1f5fbf", "-"),
          ("mean_entropy", 1, "mean entropy (spec)", "#8c8c8c", "-"),
          ("min_logprob", -1, "−min token log-prob", "#1f5fbf", ":"),
          ("max_entropy", 1, "max token entropy", "#8c8c8c", ":")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def score_answer(model, tok, question, answer):
    import torch
    prompt = tok.apply_chat_template(build_messages(question), tokenize=False,
                                     add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    answer_ids = tok(answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0].float()
    # logits at position t predict token t+1
    start = len(prompt_ids) - 1
    logp = torch.log_softmax(logits[start:start + len(answer_ids)], dim=-1)
    targets = torch.tensor(answer_ids, device=model.device)
    tok_logp = logp.gather(1, targets[:, None]).squeeze(1)
    tok_entropy = -(logp.exp() * logp).sum(-1)

    return {
        "mean_logprob": tok_logp.mean().item(),
        "mean_entropy": tok_entropy.mean().item(),
        "min_logprob": tok_logp.min().item(),
        "max_entropy": tok_entropy.max().item(),
        "n_tokens": len(answer_ids),
    }


def compute_features():
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_models("llm")
    items = pd.read_json(ITEMS_PATH, lines=True)
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.float16,
                                                 device_map="cuda").eval()
    rows = [score_answer(model, tok, r.question, r.answer)
            for r in tqdm(items.itertuples(), total=len(items), desc="Task 1 forced decoding")]
    out = pd.concat([items, pd.DataFrame(rows)], axis=1)
    out.to_csv(FEATURES, index=False)
    print(f"Computed features for {len(out)} items -> {rel(FEATURES)}")


def analyse(d):
    y = d["label"]
    print(f"\nTask 1 on all {len(d)} items (positive rate {y.mean():.2f}):")
    print(f"  {'score':28s} AUC-PR  AUROC")
    for col, sign, name, _, _ in SCORES:
        print(f"  {name:28s} {ap(y, sign * d[col]):.4f}  {roc_auc_score(y, sign * d[col]):.4f}")
    print(f"  {'answer length (reference)':28s} {ap(y, d.n_tokens):.4f}  {roc_auc_score(y, d.n_tokens):.4f}")

    cols = ["mean_logprob", "mean_entropy", "min_logprob", "max_entropy", "n_tokens"]
    print("\nMedian per class (0 = right answer, 1 = hallucinated):")
    print(d.groupby("label")[cols].median().round(3).to_string())

    print("\nOther views (AUC-PR):")
    print(f"  -sum log-prob                 : {ap(y, -d.mean_logprob * d.n_tokens):.4f}")
    print(f"  +mean log-prob (sign flipped) : {ap(y, d.mean_logprob):.4f}")
    p = d.pivot(index="pair_id", columns="label", values="mean_logprob")
    print(f"\nWithin-pair: right answer has the higher mean log-prob in {(p[0] > p[1]).mean():.0%} of pairs")


def pr_curves(d):
    fig, ax = plt.subplots(figsize=(6, 6.2))
    for col, sign, name, color, ls in SCORES:
        p, r, _ = precision_recall_curve(d.label, sign * d[col])
        ax.plot(r, p, color=color, ls=ls, lw=2, label=f"{name} (AUC-PR {ap(d.label, sign * d[col]):.3f})")
    ax.axhline(d.label.mean(), ls="--", color="#bbbbbb", label=f"Random ({d.label.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Task 1 on HaluEval QA (n={len(d)}): precision-recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true",
                        help="redo the forced-decoding pass even if task1_features.csv exists")
    args = parser.parse_args()
    log_to(OUT / "task1_results.txt")

    ensure_items()
    if args.recompute or not FEATURES.exists():
        compute_features()
    else:
        print(f"Using saved features {rel(FEATURES)} (pass --recompute to redo the GPU pass)")

    d = pd.read_csv(FEATURES)
    analyse(d)
    pr_curves(d)
    print(f"\nSaved figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
