"""Task 3: logistic regression on the Task 1 and Task 2 scores, with length checks.

Usage: python task3/task3.py   (needs the saved Task 1 and Task 2 scores)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED, TASK1_RESULTS, TASK2_RESULTS, TASK3_RESULTS, TEST_SIZE, log_to, rel

OUT = TASK3_RESULTS
T1 = ["mean_logprob", "mean_entropy"]
T2 = ["bertscore_incons", "nli_contra"]
EXTRA = ["min_logprob", "max_entropy"]
LEN = ["log_n_tokens"]
N_SPLITS = 100
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load_features():
    paths = [TASK1_RESULTS / "task1_features.csv", TASK2_RESULTS / "task2_features.csv"]
    for p in paths:
        if not p.exists():
            sys.exit(f"Missing {p}. Run task1/task1.py and task2/task2.py first.")
    t1, t2 = (pd.read_csv(p) for p in paths)
    df = t1.merge(t2[["pair_id", "label"] + T2], on=["pair_id", "label"])
    df["log_n_tokens"] = np.log(df["n_tokens"])
    return df


def split(df, seed):
    """80/20 split by question. Missing BERTScore values get the training median."""
    train_pairs, test_pairs = train_test_split(df["pair_id"].unique(), test_size=TEST_SIZE,
                                               random_state=seed)
    train = df[df["pair_id"].isin(train_pairs)].copy()
    test = df[df["pair_id"].isin(test_pairs)].sort_values(["pair_id", "label"]).copy()
    fill = train["bertscore_incons"].median()
    for part in (train, test):
        part["bertscore_incons"] = part["bertscore_incons"].fillna(fill)
    return train, test, fill


def fit_lr(train, cols):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    return clf.fit(train[cols], train["label"])


def bootstrap_ci(y, s, n=1000):
    """95% interval, resampling test questions (rows come in pairs)."""
    rng = np.random.default_rng(SEED)
    pairs = np.arange(len(y)).reshape(-1, 2)
    vals = []
    for _ in range(n):
        idx = pairs[rng.integers(0, len(pairs), len(pairs))].ravel()
        vals.append(ap(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


def pairwise_acc(test, s):
    """Share of test questions where the hallucinated answer gets the higher score."""
    p = test.assign(s=s).pivot(index="pair_id", columns="label", values="s")
    return (p[1] > p[0]).mean()


def residualise(train, test, cols):
    """Replace each feature by its residual after regressing it on log length."""
    train, test = train.copy(), test.copy()
    for c in cols:
        lr = LinearRegression().fit(train[LEN], train[c])
        train[c] -= lr.predict(train[LEN])
        test[c] -= lr.predict(test[LEN])
    return train, test


def detectors(train, test):
    ens = fit_lr(train, T1 + T2)
    return ens, {
        "Task 1: -mean log-prob": -test["mean_logprob"],
        "Task 1: mean entropy": test["mean_entropy"],
        "Task 1: LR(log-prob, entropy)": fit_lr(train, T1).predict_proba(test[T1])[:, 1],
        "Task 2: SelfCheck-BERTScore": test["bertscore_incons"],
        "Task 2: SelfCheck-NLI": test["nli_contra"],
        "Task 2: LR(BERTScore, NLI)": fit_lr(train, T2).predict_proba(test[T2])[:, 1],
        "Task 3: LR ensemble (4 features)": ens.predict_proba(test[T1 + T2])[:, 1],
        "Task 3+: LR ensemble (+ min log-prob, max entropy)":
            fit_lr(train, T1 + T2 + EXTRA).predict_proba(test[T1 + T2 + EXTRA])[:, 1],
        "Reference: answer length only": test["n_tokens"],
        "Reference: random": np.full(len(test), 0.5),
    }


def ensemble(df):
    train, test, fill = split(df, SEED)
    print(f"Split (seed={SEED}, by pair): train {len(train)} items / test {len(test)} items "
          f"({train.pair_id.nunique()} / {test.pair_id.nunique()} questions)")
    print(f"Imputed {df['bertscore_incons'].isna().sum()} missing BERTScore values "
          f"with train median {fill:.4f}")

    ens, scores = detectors(train, test)
    y = test["label"].to_numpy()
    rows = []
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        lo, hi = bootstrap_ci(y, s)
        rows.append({"detector": name, "auc_pr": ap(y, s), "ci_low": lo, "ci_high": hi,
                     "auroc": roc_auc_score(y, s)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "task3_results.csv", index=False)
    print(f"\nAUC-PR on held-out 20% (n={len(test)}, positive rate {y.mean():.2f}), "
          f"95% bootstrap CI over test pairs:")
    print(res.to_string(index=False, float_format="%.4f"))

    pred = test[["pair_id", "label"]].assign(**{k: np.asarray(v, dtype=float) for k, v in scores.items()})
    pred.to_csv(OUT / "test_predictions.csv", index=False)

    coefs = dict(zip(T1 + T2, ens[-1].coef_[0].round(4).tolist()))
    coefs["intercept"] = round(float(ens[-1].intercept_[0]), 4)
    print("\nTask 3 LR coefficients (standardised features):", coefs)
    with open(OUT / "task3_coefficients.json", "w") as f:
        json.dump(coefs, f, indent=2)
    return pred, train, test


def length_controls(train, test):
    rtrain, rtest = residualise(train, test, T1 + T2)
    scores = {
        "LR(T1)  [Task 1]": fit_lr(train, T1).predict_proba(test[T1])[:, 1],
        "LR(T2)  [Task 2]": fit_lr(train, T2).predict_proba(test[T2])[:, 1],
        "LR(T1 + T2)  [Task 3]": fit_lr(train, T1 + T2).predict_proba(test[T1 + T2])[:, 1],
        "LR(length)": fit_lr(train, LEN).predict_proba(test[LEN])[:, 1],
        "LR(length + T1)": fit_lr(train, LEN + T1).predict_proba(test[LEN + T1])[:, 1],
        "LR(length + T2)": fit_lr(train, LEN + T2).predict_proba(test[LEN + T2])[:, 1],
        "LR(length + T1 + T2)": fit_lr(train, LEN + T1 + T2).predict_proba(test[LEN + T1 + T2])[:, 1],
        "LR(T1, length removed)": fit_lr(rtrain, T1).predict_proba(rtest[T1])[:, 1],
        "LR(T2, length removed)": fit_lr(rtrain, T2).predict_proba(rtest[T2])[:, 1],
        "LR(T1 + T2, length removed)": fit_lr(rtrain, T1 + T2).predict_proba(rtest[T1 + T2])[:, 1],
    }
    for c in T1 + T2:
        sign = -1 if c == "mean_logprob" else 1
        scores[f"{c}, length removed"] = sign * rtest[c].to_numpy()
    return pd.DataFrame([{"detector": k, "auc_pr": ap(test["label"], v),
                          "pair_acc": pairwise_acc(test, v)} for k, v in scores.items()])


def length_analysis(df, train, test):
    print(f"\nLength per class (tokens): right median {df[df.label == 0].n_tokens.median():.1f}, "
          f"hallucinated median {df[df.label == 1].n_tokens.median():.1f}")
    print("Correlation of each feature with log(length), all items:")
    print(df[T1 + T2 + LEN].corr()["log_n_tokens"].drop("log_n_tokens").round(3).to_string())

    res = length_controls(train, test)
    res.to_csv(OUT / "task3_length_control.csv", index=False)
    print(f"\nSeed-{SEED} split, held-out {len(test)} items (pair_acc = share of the "
          f"{test.pair_id.nunique()} test questions ranked correctly):")
    print(res.to_string(index=False, float_format="%.4f"))

    m = fit_lr(train, LEN + T1 + T2)
    print("\nLR(length + T1 + T2) coefficients (standardised):",
          dict(zip(LEN + T1 + T2, m[-1].coef_[0].round(3).tolist())))
    rtrain, _ = residualise(train, test, T1 + T2)
    m = fit_lr(rtrain, T1 + T2)
    print("LR(T1 + T2, length removed) coefficients (standardised):",
          dict(zip(T1 + T2, m[-1].coef_[0].round(3).tolist())))

    runs = pd.concat([length_controls(*split(df, seed)[:2]).assign(seed=seed) for seed in range(N_SPLITS)])
    runs.to_csv(OUT / "task3_repeated_splits.csv", index=False)
    summary = runs.groupby("detector", sort=False)[["auc_pr", "pair_acc"]].agg(["mean", "std"])
    print(f"\nOver {N_SPLITS} random pair-level 80/20 splits (seeds 0-{N_SPLITS - 1}; mean, std):")
    print(summary.to_string(float_format="%.4f"))

    full = runs.pivot(index="seed", columns="detector", values="auc_pr")
    gain = full["LR(length + T1 + T2)"] - full["LR(length)"]
    print(f"\nAUC-PR gain of LR(length + T1 + T2) over LR(length): mean {gain.mean():+.4f}, "
          f"better on {(gain > 0).mean():.0%} of splits")
    return runs


CURVES = [("Task 1: LR(log-prob, entropy)", "Task 1 LR (log-prob, entropy)", "#eb6834", "-"),
          ("Task 2: LR(BERTScore, NLI)", "Task 2 LR (BERTScore, NLI)", "#1baf7a", "-"),
          ("Task 3: LR ensemble (4 features)", "Task 3 ensemble (4 features)", "#2a78d6", "-"),
          ("Reference: answer length only", "Answer length only", "#8c8c8c", ":")]


def pr_curves(p):
    fig, ax = plt.subplots(figsize=(6, 6.2))
    for col, name, color, ls in CURVES:
        prec, rec, _ = precision_recall_curve(p.label, p[col])
        ax.plot(rec, prec, color=color, ls=ls, lw=2, label=f"{name} (AUC-PR {ap(p.label, p[col]):.3f})")
    ax.axhline(p.label.mean(), ls="--", color="#bbbbbb", label=f"Random ({p.label.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Task 3 on held-out 20% (n={len(p)}): precision-recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=200)
    plt.close(fig)


def length_control_fig(runs):
    m = runs.groupby("detector")["auc_pr"].agg(["mean", "std"])
    groups = {"Task 1 features\n(log-prob, entropy)": ("LR(T1)  [Task 1]", "LR(T1, length removed)", "LR(length + T1)"),
              "Task 2 features\n(BERTScore, NLI)": ("LR(T2)  [Task 2]", "LR(T2, length removed)", "LR(length + T2)"),
              "Task 1 + 2 features\n(Task 3 ensemble)": ("LR(T1 + T2)  [Task 3]", "LR(T1 + T2, length removed)",
                                                        "LR(length + T1 + T2)")}
    variants = [("As is", "#8c8c8c"), ("Length removed", "#2a78d6"), ("Plus length as a feature", "#eb6834")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.26
    for j, (name, color) in enumerate(variants):
        rows = m.loc[[g[j] for g in groups.values()]]
        xs = np.arange(len(groups)) + (j - 1) * width
        bars = ax.bar(xs, rows["mean"], width - 0.02, yerr=rows["std"], color=color, label=name,
                      error_kw={"ecolor": "#444444", "elinewidth": 1, "capsize": 3}, zorder=2)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in rows["mean"]], padding=12, fontsize=9)
    length_only = m.loc["LR(length)", "mean"]
    ax.axhline(length_only, ls=":", color="#444444", zorder=1, label=f"Answer length only ({length_only:.2f})")
    ax.axhline(0.5, ls="--", color="#bbbbbb", zorder=1, label="Random (0.50)")
    ax.set_xticks(np.arange(len(groups)), list(groups))
    ax.set(ylabel=f"AUC-PR, mean ± std over {N_SPLITS} splits", ylim=(0, 1.12),
           title="Most of the signal is answer length, but some survives once it is removed")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "length_control.png", dpi=200)
    plt.close(fig)


def main():
    log_to(OUT / "task3_results.txt")
    df = load_features()
    pred, train, test = ensemble(df)
    runs = length_analysis(df, train, test)
    pr_curves(pred)
    length_control_fig(runs)
    print(f"\nSaved figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
