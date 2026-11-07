"""Task 1 — probability baseline. Run: python task1/task1.py [--recompute]

1. Forced decoding: feed [prompt + candidate answer] through the LLM once and read off, at every
   answer-token position, log p(actual token) and the entropy of the full next-token distribution.
   Averaged over answer tokens -> mean log-prob, mean entropy (plus min log-prob / max entropy).
   Hallucination score: -mean_logprob and mean_entropy (higher = more suspicious).
   Needs the GPU and models/Qwen2.5-3B-Instruct (downloaded on first use); the result is saved to
   results/task1_features.csv and reused on later runs unless --recompute is given.
2. Analysis: AUC-PR / AUROC and the answer-length check.
3. Figures: pr_curves, roc_curves, score_distributions, length_confound (.png).

Everything printed is also saved to results/task1_results.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/, for config

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from config import ITEMS_PATH, LLM_PATH, TASK1_RESULTS, build_messages, log_to, rel
from download_models import ensure_models
from prepare_data import ensure_items

OUT = TASK1_RESULTS
FEATURES = OUT / "task1_features.csv"
# (column, sign so that higher = more suspicious, label, colour, line style)
SCORES = [("mean_logprob", -1, "−mean log-prob (spec)", "#1f5fbf", "-"),
          ("mean_entropy", 1, "mean entropy (spec)", "#8c8c8c", "-"),
          ("min_logprob", -1, "−min token log-prob", "#1f5fbf", ":"),
          ("max_entropy", 1, "max token entropy", "#8c8c8c", ":")]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- 1. forced decoding

def score_answer(model, tok, question, answer):
    import torch
    prompt = tok.apply_chat_template(build_messages(question), tokenize=False,
                                     add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    answer_ids = tok(answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0].float()
    # logits at position t predict token t+1, so the answer tokens are predicted
    # by positions [len(prompt)-1, len(prompt)+len(answer)-2]
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


# ---------------------------------------------------------------- 2. analysis

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


# ---------------------------------------------------------------- 3. figures

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


def roc_curves(d):
    fig, ax = plt.subplots(figsize=(6, 6.2))
    for col, sign, name, color, ls in SCORES:
        fpr, tpr, _ = roc_curve(d.label, sign * d[col])
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
                label=f"{name} (AUROC {roc_auc_score(d.label, sign * d[col]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#bbbbbb", label="Random (0.500)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Task 1 on HaluEval QA (n={len(d)}): ROC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=200)
    plt.close(fig)


def score_distributions(d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rng = np.random.default_rng(0)  # jitter only
    for ax, (col, sign, name, color, _) in zip(axes, SCORES[:2]):
        groups = [sign * d.loc[d.label == k, col] for k in (0, 1)]
        ax.boxplot(groups, positions=[0, 1], widths=0.5, showfliers=False,
                   medianprops={"color": "black"})
        for k, g in enumerate(groups):
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(g)), g, s=8, alpha=0.35,
                       color="#1f5fbf" if k == 1 else "#8c8c8c")
        ax.set_xticks([0, 1], ["Right answer\n(label 0)", "Hallucinated\n(label 1)"])
        ax.set(title=name.replace(" (spec)", ""), ylabel="Score (higher = more suspicious)")
    fig.suptitle("Score per class: the model is more confident in hallucinated answers")
    fig.tight_layout()
    fig.savefig(OUT / "score_distributions.png", dpi=200)
    plt.close(fig)


def length_confound(d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    bins = np.arange(0, d.n_tokens.max() + 2, 2)
    for k, name, color in [(0, "Right answer", "#8c8c8c"), (1, "Hallucinated", "#1f5fbf")]:
        g = d.loc[d.label == k, "n_tokens"]
        ax1.hist(g, bins=bins, alpha=0.6, color=color, label=f"{name} (median {g.median():.1f})")
    ax1.set(xlabel="Answer length (tokens)", ylabel="Items", title="Hallucinated answers are longer")
    ax1.legend(frameon=False, fontsize=9)

    views = [(name, ap(d.label, sign * d[col])) for col, sign, name, _, _ in SCORES]
    views.append(("answer length only", ap(d.label, d.n_tokens)))
    views.sort(key=lambda v: v[1])
    colors = ["#1f5fbf" if n == "answer length only" else "#8c8c8c" for n, _ in views]
    bars = ax2.barh([n for n, _ in views], [v for _, v in views], color=colors, zorder=2)
    ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, zorder=3,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax2.axvline(0.5, ls="--", color="#bbbbbb", zorder=1)
    ax2.text(0.505, -0.6, "random", color="#888888", fontsize=9)
    ax2.set(xlabel="AUC-PR", xlim=(0, 1.08), title="Length alone beats every probability score")
    fig.tight_layout()
    fig.savefig(OUT / "length_confound.png", dpi=200)
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
    for fig in (pr_curves, roc_curves, score_distributions, length_confound):
        fig(d)
    print(f"\nSaved figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
