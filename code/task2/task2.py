"""Task 2: SelfCheckGPT (BERTScore and NLI) against 20 sampled answers per question.

Usage: python task2/task2.py [--recompute]

HaluEval has no sampled answers, so the LLM from Task 1 generates 20 per question. Scoring uses
the official selfcheckgpt package (0.1.7), with each answer treated as one sentence.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import precision_recall_curve, roc_auc_score

from config import (BERTSCORE_BASELINE, BERTSCORE_LAYER, BERTSCORE_PATH, ITEMS_PATH, LLM_PATH,
                    MAX_NEW_TOKENS, N_RESAMPLES, NLI_PATH, SAMPLES_PATH, SEED, SPACY_PATH,
                    TASK2_RESULTS, TEMPERATURE, build_messages, log_to, rel)
from download_models import ensure_models
from prepare_data import ensure_items

OUT = TASK2_RESULTS
FEATURES = OUT / "task2_features.csv"
VARIANTS = {"bertscore_incons": "SelfCheck-BERTScore", "nli_contra": "SelfCheck-NLI"}
COLORS = {"bertscore_incons": "#8c8c8c", "nli_contra": "#1f5fbf"}
KNOWN = 10  # samples (of 20) that must contain the right answer for a "known" question
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def generate_samples():
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = pd.read_json(ITEMS_PATH, lines=True)
    done = set()
    if SAMPLES_PATH.exists():
        done = {json.loads(l)["pair_id"] for l in open(SAMPLES_PATH)}
    todo = items.drop_duplicates("pair_id")
    todo = todo[~todo["pair_id"].isin(done)]
    if todo.empty:
        print(f"Using saved resamples {rel(SAMPLES_PATH)} ({len(done)} questions)")
        return False

    ensure_models("llm")
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.float16,
                                                 device_map="cuda").eval()
    with open(SAMPLES_PATH, "a") as f:
        for r in tqdm(todo.itertuples(), total=len(todo), desc="Task 2 resampling"):
            prompt = tok.apply_chat_template(build_messages(r.question), tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            torch.manual_seed(SEED + int(r.pair_id))
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE,
                                     top_p=1.0, top_k=0, repetition_penalty=1.0,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     num_return_sequences=N_RESAMPLES,
                                     pad_token_id=tok.eos_token_id)
            samples = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"pair_id": int(r.pair_id),
                                "samples": [s.strip() for s in samples]}) + "\n")
            f.flush()
    print(f"Wrote resamples for {len(todo)} questions -> {rel(SAMPLES_PATH)}")
    del model
    torch.cuda.empty_cache()
    return True


def patch_official_code():
    """Make selfcheckgpt 0.1.7 run with transformers 5 and load its models from models/."""
    import bert_score
    import spacy
    from transformers import PreTrainedTokenizerBase

    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        # removed in transformers 5; SelfCheckNLI only passes (text, text_pair) tuples
        first, second = zip(*batch_text_or_text_pairs)
        return self(list(first), list(second), **kwargs)

    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        PreTrainedTokenizerBase.batch_encode_plus = batch_encode_plus

    spacy_load = spacy.load
    spacy.load = lambda name, *a, **k: spacy_load(SPACY_PATH if name == "en_core_web_sm" else name, *a, **k)

    # bert_score.score() reloads roberta-large on every call, so keep one scorer
    scorers = {}

    def cached_score(cands, refs, lang=None, rescale_with_baseline=False, verbose=False):
        assert lang == "en", lang
        if rescale_with_baseline not in scorers:
            scorers[rescale_with_baseline] = bert_score.BERTScorer(
                model_type=str(BERTSCORE_PATH), num_layers=BERTSCORE_LAYER, lang=lang,
                rescale_with_baseline=rescale_with_baseline, baseline_path=str(BERTSCORE_BASELINE))
        return scorers[rescale_with_baseline].score(cands, refs, verbose=verbose)

    bert_score.score = cached_score


def score_selfcheck():
    ensure_models("bertscore", "nli", "spacy")
    patch_official_code()
    import torch
    from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNLI
    from tqdm import tqdm

    items = pd.read_json(ITEMS_PATH, lines=True)
    samples = {d["pair_id"]: d["samples"] for d in map(json.loads, open(SAMPLES_PATH))}
    bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    nli = SelfCheckNLI(nli_model=str(NLI_PATH), device=torch.device("cuda"))

    # SelfCheckBERTScore fails on samples with no sentence longer than 3 tokens
    usable = {pid: [s for s in ss if any(len(sent) > 3 for sent in bertscore.nlp(s).sents)]
              for pid, ss in samples.items()}

    rows = []
    for r in tqdm(items.itertuples(), total=len(items), desc="Task 2 official SelfCheckGPT"):
        u = usable[r.pair_id]
        b = bertscore.predict(sentences=[r.answer], sampled_passages=u)[0] if u else np.nan
        n = nli.predict(sentences=[r.answer], sampled_passages=samples[r.pair_id])[0]
        rows.append({"bertscore_incons": b, "nli_contra": n, "n_bertscore_samples": len(u)})

    out = pd.concat([items, pd.DataFrame(rows)], axis=1)
    out.to_csv(FEATURES, index=False)
    print(f"Scored {len(out)} items -> {rel(FEATURES)}")


def knowledge_hits(d):
    """Number of samples per question that contain the right answer."""
    samples = {r["pair_id"]: r["samples"] for r in map(json.loads, open(SAMPLES_PATH))}
    right = d[d.label == 0].set_index("pair_id")["answer"]
    return pd.Series({pid: sum(a.lower().strip(" .") in s.lower() for s in samples[pid])
                      for pid, a in right.items()})


def analyse(d):
    y = d["label"]
    n_used = d.drop_duplicates("pair_id")["n_bertscore_samples"]
    print(f"\nBERTScore samples usable per question: mean {n_used.mean():.1f}/20, "
          f"min {n_used.min()}, questions with none: {(n_used == 0).sum()}")

    print(f"\nTask 2 (official selfcheckgpt code), positive rate {y.mean():.2f}:")
    print(f"  {'variant':22s} AUC-PR  AUROC   n")
    for c, name in VARIANTS.items():
        s = d.dropna(subset=[c])
        print(f"  {name:22s} {ap(s.label, s[c]):.4f}  {roc_auc_score(s.label, s[c]):.4f}  {len(s)}")

    s = d.dropna(subset=["bertscore_incons"])
    cols = list(VARIANTS)
    print("\nMedian score per class (0 = right answer, 1 = hallucinated; higher = more suspicious):")
    print(s.groupby("label")[cols].median().round(3).to_string())
    print("\nWithin-pair: hallucinated answer gets the higher score in")
    for c in cols:
        p = s.pivot(index="pair_id", columns="label", values=c)
        print(f"  {c:22s}: {(p[1] > p[0]).mean():.0%} of pairs")
    print("\nSign-flipped AUC-PR (the score used in reverse):")
    for c in cols:
        print(f"  -{c:21s}: {ap(s.label, -s[c]):.4f}")

    hits = knowledge_hits(d)
    print(f"\nSamples containing the right answer (substring match), over {len(hits)} questions:")
    print(f"  >=1 of 20 samples : {(hits > 0).mean():.0%} of questions")
    print(f"  >={KNOWN} of 20 samples: {(hits >= KNOWN).mean():.0%} of questions")
    print(f"  mean per question : {hits.mean():.1f} / 20")
    known = d.pair_id.isin(hits[hits >= KNOWN].index)
    for name, mask in [(f"known (>={KNOWN}/20)", known), ("not known", ~known)]:
        sub = d[mask]
        sb = sub.dropna(subset=["bertscore_incons"])
        print(f"  AUC-PR on {name:16s} ({sub.pair_id.nunique()} questions): "
              f"NLI {ap(sub.label, sub.nli_contra):.4f}, BERTScore {ap(sb.label, sb.bertscore_incons):.4f}")


def pr_curves(d):
    fig, ax = plt.subplots(figsize=(6, 6.2))
    for col, name in VARIANTS.items():
        s = d.dropna(subset=[col])
        p, r, _ = precision_recall_curve(s.label, s[col])
        ax.plot(r, p, color=COLORS[col], lw=2,
                label=f"{name} (AUC-PR {ap(s.label, s[col]):.3f}, n={len(s)})")
    ax.axhline(d.label.mean(), ls="--", color="#bbbbbb", label=f"Random ({d.label.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="SelfCheckGPT on HaluEval QA: precision-recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=200)
    plt.close(fig)


def auc_by_knowledge(d):
    hits = knowledge_hits(d)
    known = set(hits[hits >= KNOWN].index)
    subsets = {f"Model knows it\n(>={KNOWN}/20 samples right,\n{len(known)} questions)": d.pair_id.isin(known),
               f"Model mostly doesn't\n({len(hits) - len(known)} questions)": ~d.pair_id.isin(known),
               "All questions": d.pair_id.notna()}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.36
    for j, (col, name) in enumerate(VARIANTS.items()):
        vals = []
        for mask in subsets.values():
            s = d[mask].dropna(subset=[col])
            vals.append(ap(s.label, s[col]))
        xs = np.arange(len(subsets)) + (j - 0.5) * width
        bars = ax.bar(xs, vals, width, color=COLORS[col], label=name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=10)
    ax.axhline(0.5, ls="--", color="#bbbbbb", label="Random (0.50)")
    ax.set_xticks(np.arange(len(subsets)), list(subsets))
    ax.set(ylabel="AUC-PR", ylim=(0, 1.05), title="SelfCheck-NLI works when the LLM knows the answer")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "auc_by_knowledge.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true",
                        help="regenerate the resamples and rescore, even if saved results exist")
    args = parser.parse_args()
    log_to(OUT / "task2_results.txt")

    ensure_items()
    if args.recompute:
        SAMPLES_PATH.unlink(missing_ok=True)
        FEATURES.unlink(missing_ok=True)
    if generate_samples():
        FEATURES.unlink(missing_ok=True)
    if not FEATURES.exists():
        score_selfcheck()
    else:
        print(f"Using saved scores {rel(FEATURES)} (pass --recompute to redo)")

    d = pd.read_csv(FEATURES)
    analyse(d)
    pr_curves(d)
    auc_by_knowledge(d)
    print(f"\nSaved figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
