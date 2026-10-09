"""Analysis for cloze and Likert/forced-choice outputs.

Usage:
    python analyze.py --cloze cloze_*.csv --likert likert_*.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def analyze_cloze(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Average over templates within item/trait/variety/model.
    g = (df.groupby(["model", "item_id", "trait", "variety"], as_index=False)["shift"].mean())
    wide = g.pivot_table(index=["model", "item_id", "trait"], columns="variety", values="shift").reset_index()
    wide["diff"] = wide["pidgin"] - wide["british"]  # >0: trait more likely for Pidgin text

    print("\n== Cloze: mean shift (Pidgin - British), per model and trait ==")
    summ = (wide.groupby(["model", "trait"])["diff"]
            .agg(["mean", "std", "count"]).reset_index())
    summ["t_p"] = [
        stats.ttest_1samp(wide[(wide.model == m) & (wide.trait == t)]["diff"], 0).pvalue
        for m, t in zip(summ["model"], summ["trait"])
    ]
    # Holm correction across traits within model
    summ["t_p_holm"] = summ.groupby("model")["t_p"].transform(
        lambda p: np.minimum(1, (p.rank(method="first").rsub(len(p) + 1)) * p)
    )
    print(summ.sort_values(["model", "mean"]).to_string(index=False))

    # Mixed model: shift ~ variety * trait, random intercept for item (per model).
    for m, sub in g.groupby("model"):
        if sub["item_id"].nunique() < 5:
            print(f"[{m}] too few items for a mixed model")
            continue
        fit = smf.mixedlm("shift ~ C(variety) * C(trait)", sub, groups=sub["item_id"]).fit()
        print(f"\n[{m}] mixed model, variety main effect:")
        print(fit.summary().tables[1].loc[lambda t: t.index.str.contains("variety")])


def analyze_likert(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.dropna(subset=["value"])

    lik = df[df.task == "likert"].copy()
    if not lik.empty:
        lik = lik.groupby(["model", "item_id", "trait", "variety", "condition"], as_index=False)["value"].mean()
        print("\n== Likert: Pidgin vs British (mean over repeats) ==")
        out = []
        for (m, t, c), sub in lik.groupby(["model", "trait", "condition"]):
            w = sub.pivot(index="item_id", columns="variety", values="value").dropna()
            if len(w) < 5:
                continue
            p = stats.wilcoxon(w["pidgin"], w["british"]).pvalue if (w["pidgin"] != w["british"]).any() else 1.0
            out.append({"model": m, "trait": t, "condition": c, "n": len(w),
                        "mean_pidgin": w["pidgin"].mean(), "mean_british": w["british"].mean(),
                        "cliffs_delta": cliffs_delta(w["pidgin"], w["british"]), "wilcoxon_p": p})
        print(pd.DataFrame(out).to_string(index=False))

    fc = df[df.task == "forced_choice"]
    if not fc.empty:
        print("\n== Forced choice: share choosing the Pidgin writer ==")
        for (m, t), sub in fc.groupby(["model", "trait"]):
            k, n = int(sub["value"].sum()), len(sub)
            p = stats.binomtest(k, n, 0.5).pvalue
            print(f"{m:30s} {t:14s} {k}/{n} = {k/n:.2f}  (binomial p={p:.3g})")
        print("Note: repeats of the same item are not independent; prefer a logistic mixed model for inference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloze", nargs="*", default=[])
    ap.add_argument("--likert", nargs="*", default=[])
    a = ap.parse_args()
    if a.cloze:
        analyze_cloze([p for pat in a.cloze for p in glob.glob(pat)])
    if a.likert:
        analyze_likert([p for pat in a.likert for p in glob.glob(pat)])