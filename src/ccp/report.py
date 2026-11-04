"""Markdown report + figures for the journal appendix."""

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


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if "index" not in df.columns else df.to_markdown()


def _figures(out: Path, metrics: dict, results: pd.DataFrame, text_table: pd.DataFrame, dcr_dists: dict) -> None:
    fig_dir = out / "figures"
    uniq = metrics["uniqueness_by_qi_set"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(list(uniq), [100 * v for v in uniq.values()], color="#4c72b0")
    ax.set_ylabel("students unique (%)")
    ax.set_title("How few attributes single out a student at Campus Cham")
    for i, v in enumerate(uniq.values()):
        ax.text(i, 100 * v + 1, f"{100 * v:.0f} %", ha="center")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fig_dir / "1_uniqueness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    t = text_table.sort_values("recall")
    ax.barh(t.index, 100 * t["recall"], color="#55a868", label="detected")
    ax.barh(t.index, 100 * t["leak_rate"], color="#c44e52", label="left in text (leak)")
    ax.set_xlabel("% of ground-truth spans")
    ax.set_title("Presidio on German + English tickets")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "2_text_detection.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(dcr_dists), figsize=(4 * len(dcr_dists), 3.2), sharey=True)
    for ax, (name, (d_h, d_s)) in zip(np.atleast_1d(axes), dcr_dists.items()):
        bins = np.linspace(0, max(d_h.max(), d_s.max()), 30)
        ax.hist(d_h, bins=bins, alpha=0.6, density=True, label="real, not in training")
        ax.hist(d_s, bins=bins, alpha=0.6, density=True, label="synthetic")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("distance to closest training record")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "3_distance_to_closest_record.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(results))
    ax1.bar(x - 0.2, 100 * results["belief_if_true"], 0.4, color="#c44e52",
            label="attacker's certainty a unique student gets disability support (%)")
    ax1.bar(x + 0.2, 100 * results["ml_auc_at_risk"], 0.4, color="#4c72b0", label="usefulness: 'at risk' model AUC x100")
    ax1.axhline(100 * results["prior"].iloc[0], color="#c44e52", ls="--", lw=1, label="attacker's prior (base rate)")
    ax1.axhline(50, color="#4c72b0", ls=":", lw=1, label="AUC of random guessing")
    ax1.set_xticks(x, results["release"], rotation=15, ha="right", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("Privacy vs usefulness per release")
    fig.tight_layout()
    fig.savefig(fig_dir / "4_privacy_vs_utility.png", dpi=160)
    plt.close(fig)


def write_report(out, metrics, results, text_table, misses, pseudo, question, question_lvl, dcr_dists, students, ka, hier) -> None:
    _figures(out, metrics, results, text_table, dcr_dists)
    cfg, ts = metrics["config"], metrics["text"]
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

    add("## 2. Free text: Presidio masking (German + English)\n")
    add(f"- Ground-truth personal-data spans: {ts['spans']}\n"
        f"- Direct identifiers detected: **{_pct(ts['recall_direct_identifiers'])}**; fully masked: "
        f"{_pct(ts['fully_masked_direct_identifiers'])}; still readable verbatim: **{_pct(ts['leak_rate_direct_identifiers'])}**\n"
        f"- Tickets with no direct identifier left: {_pct(ts['tickets_without_direct_leak'])}\n"
        f"- Indirect identifiers (e.g. 'the only student from Ghana in MRO'): on average only "
        f"{_pct(ts['indirect_identifiers_mean_coverage'])} of the phrase gets masked — NER is not built for them\n"
        f"- Precision (detections that really were personal data): {_pct(ts['precision'])}\n"
        f"- By language: recall EN {_pct(ts.get('recall_en', float('nan')))}, DE {_pct(ts.get('recall_de', float('nan')))}\n")
    tt = text_table.copy()
    for c in ["recall", "coverage", "leak_rate"]:
        tt[c] = tt[c].map(_pct)
    add(tt.to_markdown())
    add("\n![text](figures/2_text_detection.png)\n")
    add("### Examples: keyed pseudonyms (the same string always maps to the same token)\n")
    for r in pseudo.head(4).itertuples():
        add(f"- original: {r.text}\n  - pseudonymised: {r.text_anonymised}")
    add("")
    if len(misses):
        add("### Sample of misses (not fully masked)\n")
        sample = misses[misses.label != "INDIRECT_IDENTIFIER"].groupby("label").head(2)[["label", "lang", "coverage", "pii_text", "after"]]
        add(_md(sample))
        add(f"\nAll {len(misses)} partially or fully missed spans: `text_misses.csv`.\n")

    add("## 3. Tables: k-anonymity with l-diversity\n")
    k_m = metrics["k_anonymity"]
    add(f"Searched {k_m['nodes_checked']} generalisation combinations, {k_m['nodes_feasible']} satisfied "
        f"k={cfg['k']} and l={cfg['l']} (on disability support) with at most {cfg['max_suppression']:.0%} rows suppressed. "
        f"Chosen (lowest information loss {k_m['loss']:.2f}, {k_m['suppressed']} rows suppressed, {k_m['classes']} groups):\n")
    add(_md(pd.DataFrame({"attribute": list(k_m["node"]),
                          "released as": [LEVEL_NAMES[q][lvl] for q, lvl in k_m["node"].items()]})))
    add("\nSample of the released table:\n")
    add(_md(ka.data.head(5)))
    add("")

    add("## 4. Every release compared\n")
    add("Privacy: *linkage* = share of students who are unique in the real data and can be found in the release by "
        "their quasi-identifiers; *belief* = the attacker's average certainty that such a student receives disability "
        f"support, for students who really do (prior: {_pct(results['prior'].iloc[0])}); *certain* = students whose "
        "disability status is revealed with 100 % certainty. Synthetic rows are nobody, so for twins we also check "
        "*exact copies* of training rows and the distance to the closest record (DCR) against real rows the model never saw.\n")
    add("Utility: *marginals* = 1 − total variation distance per column (1 = identical); *assoc. gap* = mean change of "
        "pairwise Cramér's V (0 = relationships kept); *AUC* = model trained on the release predicting 'at risk' "
        "for real unseen students (0.5 = useless).\n")
    view = pd.DataFrame({
        "release": results["release"],
        "rows": results["rows"],
        "unique": results.get("unique_share", pd.Series(dtype=float)).map(_pct),
        "linkage": results["linkage_matched"].map(_pct),
        "belief": results["belief_if_true"].map(_pct),
        "certain": results["certain_disclosures"],
        "exact copies": results.get("exact_copies", pd.Series(dtype=float)).map(_pct),
        "marginals": results["marginal_similarity"].round(3),
        "assoc. gap": results["association_gap"].round(3),
        "AUC": results["ml_auc_at_risk"].round(3),
    })
    add(_md(view))
    add("\n![privacy vs utility](figures/4_privacy_vs_utility.png)\n")
    add("![dcr](figures/3_distance_to_closest_record.png)\n")

    add("## 5. A real analyst question: average grade by programme\n")
    add(_md(question.reset_index()))
    add("\nOn the k-anonymised level (programme generalised as chosen above):\n")
    add(_md(question_lvl.reset_index()))
    add("")

    add("## 6. Runtime (laptop CPU)\n")
    add(_md(pd.DataFrame({"step": list(metrics["timings_s"]), "seconds": list(metrics["timings_s"].values())})))
    add("")
    (out / "report.md").write_text("\n".join(L))
