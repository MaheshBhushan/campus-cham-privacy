"""Markdown report + figures for the journal appendix. Every number comes from the run's metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVEL_NAMES = {
    "age": ["exact", "5-year band", "10-year band", "removed"],
    "gender": ["exact", "removed"],
    "nationality": ["country", "world region", "Germany / international", "removed"],
    "programme": ["programme", "Bachelor / Master", "removed"],
    "enrolment_year": ["exact", "2-year band", "removed"],
    "postcode": ["5 digits", "3 digits", "1 digit", "removed"],
}


def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.1f} %"


def _md(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figures(fig_dir: Path, metrics: dict, results: pd.DataFrame, text_table: pd.DataFrame, dcr_dists: dict,
             sweep: pd.DataFrame) -> None:
    uniq = metrics["uniqueness_by_qi_set"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(list(uniq), [100 * v for v in uniq.values()], color="#4c72b0")
    ax.set_ylabel("students unique (%)")
    ax.set_title("How few attributes single out a student at Campus Cham")
    for i, v in enumerate(uniq.values()):
        ax.text(i, 100 * v + 1, f"{100 * v:.0f} %", ha="center")
    ax.set_ylim(0, 105)
    _save(fig, fig_dir / "1_uniqueness.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    t = text_table.sort_values("recall")
    ax.barh(t.index, 100 * t["coverage"], color="#55a868", label="share of characters masked")
    ax.barh(t.index, 100 * t["leak_rate"], color="#c44e52", label="still readable verbatim")
    ax.set_xlabel("% of ground-truth spans")
    ax.set_title("Presidio + register on German and English tickets")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, fig_dir / "2_text_detection.png")

    fig, axes = plt.subplots(1, len(dcr_dists), figsize=(3.2 * len(dcr_dists), 3.2), sharey=True)
    for ax, (name, (d_h, d_s)) in zip(np.atleast_1d(axes), dcr_dists.items()):
        bins = np.linspace(0, max(d_h.max(), d_s.max()), 30)
        ax.hist(d_h, bins=bins, alpha=0.6, density=True, label="real, not in training")
        ax.hist(d_s, bins=bins, alpha=0.6, density=True, label="synthetic")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("distance to closest training record", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, fig_dir / "3_distance_to_closest_record.png")

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (col, label, color) in enumerate([
        ("singling_out", "singling out", "#c44e52"), ("linkability", "linkability", "#dd8452"),
        ("inference", "inference (disability)", "#8172b3"), ("mia_auc_excess", "membership (AUC-0.5)x2", "#937860"),
    ]):
        vals = results[col] if col != "mia_auc_excess" else (results["mia_auc"] - 0.5).clip(lower=0) * 2
        ax.bar(x + (i - 1.5) * 0.2, 100 * vals, 0.2, color=color, label=label)
    ax.set_xticks(x, results["release"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("attack success above baseline (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("The three GDPR anonymisation tests (WP29 05/2014) + membership inference")
    _save(fig, fig_dir / "4_gdpr_criteria.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, 100 * results["belief_if_true"], 0.4, color="#c44e52",
           label="attacker's certainty a unique student gets disability support (%)")
    ax.bar(x + 0.2, 100 * results["ml_auc_at_risk"], 0.4, color="#4c72b0", label="usefulness: 'at risk' model AUC x100")
    ax.axhline(100 * results["prior"].iloc[0], color="#c44e52", ls="--", lw=1, label="attacker's prior (base rate)")
    ax.axhline(50, color="#4c72b0", ls=":", lw=1, label="AUC of random guessing")
    ax.set_xticks(x, results["release"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Privacy vs usefulness per release")
    _save(fig, fig_dir / "5_privacy_vs_utility.png")

    if len(sweep):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        ref = results.set_index("release")
        for model, g in sweep.groupby("model"):
            axes[0].plot(g.epsilon, g.ml_auc_at_risk, marker="o", label=model.upper())
            axes[1].plot(g.epsilon, g.marginal_similarity, marker="o", label=model.upper())
        if "twin (GaussianCopula)" in ref.index:
            axes[0].axhline(ref.loc["twin (GaussianCopula)", "ml_auc_at_risk"], color="grey", ls="--", label="non-DP copula twin")
            axes[1].axhline(ref.loc["twin (GaussianCopula)", "marginal_similarity"], color="grey", ls="--", label="non-DP copula twin")
        axes[0].axhline(ref.loc["minimised raw", "ml_auc_at_risk"], color="black", ls=":", label="real data")
        for ax, title in zip(axes, ["usefulness: AUC on real unseen students", "marginal similarity (1 = identical)"]):
            ax.set_xscale("log")
            ax.set_xlabel("ε (smaller = stronger privacy)")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
        _save(fig, fig_dir / "6_epsilon_sweep.png")


def write_report(out: Path, metrics: dict, results: pd.DataFrame, text: dict, kanon_cmp: pd.DataFrame, mo,
                 question: pd.DataFrame, dcr_dists: dict, sweep: pd.DataFrame) -> None:
    _figures(out / "figures", metrics, results, text["table"], dcr_dists, sweep)
    cfg, ts, ind = metrics["config"], metrics["text"], metrics["indirect"]
    L: list[str] = []
    add = L.append
    add("# Campus Cham confidential data pipeline: results\n")
    add(f"All data is invented ({cfg['n_students']} students, {cfg['n_tickets']} tickets, seed {cfg['seed']}); "
        "distributions are calibrated to public statistics (see `docs/calibration.md`). "
        f"The student table was split into {cfg['train_rows']} rows used for release and {cfg['holdout_rows']} "
        "held-out rows that act as 'real people the model never saw'.\n")
    add("Regenerate with `uv run ccp run` — every number below comes from `metrics.json`.\n")

    add("## 1. Why a small campus is risky\n")
    add("Share of students who are the only person with that combination of attributes (in the minimised table, "
        "i.e. after names, IDs, contact data and birth dates were already removed):\n")
    add(_md(pd.DataFrame({"attributes known to an attacker": list(metrics["uniqueness_by_qi_set"]),
                          "students unique": [_pct(v) for v in metrics["uniqueness_by_qi_set"].values()]})))
    add("\n![uniqueness](figures/1_uniqueness.png)\n")

    add("## 2. Free text\n")
    add("### 2.1 Direct identifiers: three detection pipelines\n")
    add("*register* = names from the university's own enrolment and staff register used as a deny-list "
        "(the university already holds them; nothing new is collected).\n")
    v = text["variants"].copy()
    for c in ["recall_direct_identifiers", "leak_rate_direct_identifiers", "person_leak_rate", "precision",
              "indirect_identifiers_mean_coverage"]:
        v[c] = v[c].map(_pct)
    v = v.rename(columns={"recall_direct_identifiers": "detected", "leak_rate_direct_identifiers": "readable verbatim",
                          "person_leak_rate": "names readable", "indirect_identifiers_mean_coverage": "indirect phrases masked"})
    secs = {n: metrics["timings_s"].get(f"text: {n}") for n in text["variants"].pipeline}
    v["seconds"] = v["pipeline"].map(secs)
    add(_md(v))
    add(f"\nDetail for the pipeline used downstream (**{text['main']}**), {ts['spans']} ground-truth spans:\n")
    tt = text["table"].copy()
    for c in ["recall", "coverage", "leak_rate"]:
        tt[c] = tt[c].map(_pct)
    add(_md(tt, index=True))
    add("\n![text](figures/2_text_detection.png)\n")
    add("### 2.2 Indirect identifiers ('the only student from Ghana in MRO')\n")
    add("NER does not understand that age + origin + programme together identify someone. A rule-based step "
        "coarsens self-descriptions after Presidio (country → world region, age → decade, programme → degree, "
        "gender words and 'the only' removed). *Anonymity set* = how many students match what the text still reveals.\n")
    add(_md(pd.DataFrame({
        "": ["after Presidio + register", "after indirect-identifier step"],
        "texts with a self-description": [ind["texts"], ind["texts"]],
        "writer unique (set = 1)": [_pct(ind["unique_before"]), _pct(ind["unique_after"])],
        "median anonymity set": [ind["median_before"], ind["median_after"]],
        "smallest set": [ind["min_before"], ind["min_after"]],
    })))
    add("\nExamples:\n")
    for r in text["indirect_examples"].itertuples():
        add(f"- [{r.lang}] {r.text_anonymised}\n  - → {r.text_final}")
    add("\nA local LLM rewrite (Qwen2.5-1.5B on CPU) was tested as an alternative: see `docs/llm_rewrite.md`.\n")
    add("### 2.3 Keyed pseudonyms (the same string always maps to the same token)\n")
    for r in text["pseudo"].head(3).itertuples():
        add(f"- original: {r.text}\n  - pseudonymised: {r.text_anonymised}")
    misses = text["misses"]
    direct_miss = misses[misses.label != "INDIRECT_IDENTIFIER"] if len(misses) else misses
    if len(direct_miss):
        add("\n### 2.4 Remaining misses of direct identifiers\n")
        add(_md(direct_miss.groupby("label").head(3)[["label", "lang", "coverage", "pii_text"]]))
        add(f"\nAll {len(misses)} partially or fully missed spans (incl. indirect phrases): `text_misses.csv`.\n")
    add("")

    add("## 3. Tables: k-anonymity with l-diversity\n")
    k_m = metrics["k_anonymity"]
    add(f"Goal: every student shares their quasi-identifiers with at least k={cfg['k']} others, and every such group "
        f"contains students with and without disability support (l={cfg['l']}).\n")
    add(f"**Full-domain** (ARX-style): searched {k_m['nodes_checked']} combinations of one generalisation level per "
        f"attribute, {k_m['nodes_feasible']} were valid with ≤ {cfg['max_suppression']:.0%} suppression; best:\n")
    add(_md(pd.DataFrame({"attribute": list(k_m["node"]),
                          "released as": [LEVEL_NAMES[q][lvl] for q, lvl in k_m["node"].items()]})))
    add("\n**Mondrian** (local recoding): splits the students into groups top-down, so each group gets its own level "
        "of detail; a large group of Indian Master's students can keep 'South Asia' while a rare nationality "
        "falls back to 'International'.\n")
    kc = kanon_cmp.copy()
    kc["loss_ncp"] = kc["loss_ncp"].round(3)
    kc["nationality_kept"] = kc["nationality_kept"].map(_pct)
    add(_md(kc.rename(columns={"loss_ncp": "information loss (NCP, 0 = raw)", "nationality_kept": "nationality kept as country/region"})))
    add("\n(The last row also generalises grades, credits and semester, so its loss is measured over more columns.)\n")
    add("Mondrian release, first rows:\n")
    add(_md(mo.data.head(5)))
    add("")

    add("## 4. Every release attacked\n")
    add("**GDPR test** (Article 29 Working Party, Opinion 05/2014: data is anonymous only if no one can *single out* "
        "a person, *link* records about them, or *infer* facts about them), measured with Anonymeter: risk = attack "
        "success above a random-guess baseline, normalised (0 = no better than guessing, 1 = always right). "
        "Real held-out students act as a control, so the risk reflects what the release leaks rather than what "
        "can be guessed about anyone. *Membership* = can an attacker tell whether a student was in the data (AUC 0.5 = no).\n")
    add("**Usefulness**: *marginals* = 1 − total variation distance per column; *assoc. gap* = mean change of "
        "pairwise Cramér's V; *AUC* = model trained on the release predicting 'at risk' for real unseen students.\n")
    view = pd.DataFrame({
        "release": results["release"],
        "unique": results.get("unique_share", pd.Series(dtype=float)).map(_pct),
        "singling out": results["singling_out"].round(2),
        "linkability": results["linkability"].round(2),
        "inference": results["inference"].round(2),
        "membership AUC": results["mia_auc"].round(2),
        "certain disclosures": results["certain_disclosures"],
        "exact copies": results.get("exact_copies", pd.Series(dtype=float)).map(_pct),
        "marginals": results["marginal_similarity"].map(lambda x: "–" if pd.isna(x) else f"{x:.3f}"),
        "assoc. gap": results["association_gap"].map(lambda x: "–" if pd.isna(x) else f"{x:.3f}"),
        "AUC": results["ml_auc_at_risk"].round(3),
    })
    add(_md(view))
    add("\n![gdpr](figures/4_gdpr_criteria.png)\n")
    add("![privacy vs utility](figures/5_privacy_vs_utility.png)\n")
    add("![dcr](figures/3_distance_to_closest_record.png)\n")
    if metrics["attack_notes"]:
        add("Attack caveats reported by the evaluator ('as good or worse as baseline' = the attack learned "
            "nothing, reported as risk 0):\n")
        for name, notes in metrics["attack_notes"].items():
            for n in notes:
                add(f"- *{name}*: {n}")
        add("")

    if len(sweep):
        add("## 5. Differential privacy: how much privacy costs\n")
        add("ε bounds how much any single student can change the output: with ε = 1 an attacker's odds about any fact "
            "on one student rise by at most a factor e ≈ 2.7 (except with probability δ = 10⁻⁹), whatever else they "
            "know. Smaller ε = stronger guarantee, noisier data. All bins and category lists come from public sources, "
            "so the guarantee covers the whole release.\n")
        s = sweep[["model", "epsilon", "marginal_similarity", "association_gap", "ml_auc_at_risk", "belief_if_true", "runtime_s"]].copy()
        s["belief_if_true"] = s["belief_if_true"].map(_pct)
        add(_md(s.round(3)))
        add("\n![sweep](figures/6_epsilon_sweep.png)\n")

    n = 6 if len(sweep) else 5
    add(f"## {n}. A real analyst question: average grade by programme\n")
    add(_md(question, index=True))
    add("")

    add(f"## {n + 1}. Runtime (laptop CPU, 8 cores)\n")
    add(_md(pd.DataFrame({"step": list(metrics["timings_s"]), "seconds": list(metrics["timings_s"].values())})))
    add("")
    (out / "report.md").write_text("\n".join(L))
