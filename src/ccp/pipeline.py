"""End-to-end run: generate -> minimise -> mask text -> k-anonymise -> synthesise -> evaluate -> report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as C
from . import evaluate as E
from .generate import DIRECT_IDENTIFIERS, QUASI_IDENTIFIERS, generate_students, generate_tickets
from .kanon import anonymise, build_hierarchies, generalise, reid_risk
from .report import write_report
from .synth import synthesise
from .textmask import anonymise_tickets

TARGET = "at_risk"            # analysis target for the ML-utility test
SENSITIVE = "disability_support"  # GDPR Art. 9 attribute protected by l-diversity, attacked by linkage


def run(
    out: Path,
    n_students: int = C.CAMPUS_CHAM_STUDENTS,
    n_tickets: int = 600,
    seed: int = 42,
    k: int = 5,
    l: int = 2,
    max_suppression: float = 0.05,
    ctgan_epochs: int = 300,
    pseudonym_key: bytes = b"demo-key-kept-by-the-data-protection-officer",
) -> dict:
    data_dir, fig_dir = out / "data", out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name, fn, *a, **kw):
        t = time.perf_counter()
        r = fn(*a, **kw)
        timings[name] = round(time.perf_counter() - t, 2)
        print(f"  {name}: {timings[name]} s", flush=True)
        return r

    print("1/6 generating fake Campus Cham data", flush=True)
    students = timed("generate_students", generate_students, n_students, seed)
    tickets = timed("generate_tickets", generate_tickets, students, n_tickets, seed)
    students.to_csv(data_dir / "students_raw.csv", index=False)
    tickets.to_csv(data_dir / "tickets_raw.csv", index=False)

    # Data minimisation (GDPR Art. 5(1)(c)): analysts never need names, contact data or IDs.
    research = students.drop(columns=DIRECT_IDENTIFIERS + ["degree"])  # degree is implied by programme
    # Own RNG stream: reusing the generator's seed would make the split correlate with programme.
    rng = np.random.default_rng([seed, 2])
    is_train = rng.random(len(research)) < 0.8
    train, holdout = research[is_train].reset_index(drop=True), research[~is_train].reset_index(drop=True)
    train.to_csv(data_dir / "students_minimised.csv", index=False)

    print("2/6 masking personal data in free text (Presidio)", flush=True)
    masked = timed("presidio_mask", anonymise_tickets, tickets, "mask")
    masked.to_csv(data_dir / "tickets_masked.csv", index=False)
    pseudo = timed("presidio_pseudonymise", anonymise_tickets, tickets.head(8), "pseudonym", pseudonym_key)
    text_table, text_summary, misses = E.text_scores(masked)
    misses.to_csv(out / "text_misses.csv", index=False)

    print(f"3/6 k-anonymity (k={k}, l={l} on {SENSITIVE}, <= {max_suppression:.0%} suppression)", flush=True)
    region = {**C.REGION}
    prog_level = {p: v[1] for p, v in C.PROGRAMMES.items()}
    hier = build_hierarchies(region, prog_level)
    ka = timed("k_anonymity", anonymise, train, hier, k=k, max_suppression=max_suppression, sensitive=SENSITIVE, l=l)
    ka.data.to_csv(data_dir / "students_k_anonymised.csv", index=False)
    train_view = generalise(train, hier, ka.node)
    holdout_view = generalise(holdout, hier, ka.node)

    print("4/6 synthetic twins (SDV)", flush=True)
    twins = {
        "twin_of_k_anon_gc": timed("twin_k_anon_gaussian_copula", synthesise, ka.data, "gaussian_copula", len(train), seed),
        "twin_of_raw_gc": timed("twin_raw_gaussian_copula", synthesise, train, "gaussian_copula", len(train), seed),
        "twin_of_raw_ctgan": timed("twin_raw_ctgan", synthesise, train, "ctgan", len(train), seed, ctgan_epochs),
    }
    for name, df in twins.items():
        df.to_csv(data_dir / f"students_{name}.csv", index=False)

    print("5/6 evaluating privacy and utility", flush=True)
    qis = QUASI_IDENTIFIERS
    uniq = {}
    for label, cols in {
        "nationality + programme": ["nationality", "programme"],
        "+ gender": ["nationality", "programme", "gender"],
        "+ age": ["nationality", "programme", "gender", "age"],
        "all 6 QIs": qis,
    }.items():
        uniq[label] = reid_risk(train, cols)["unique_share"]

    releases = {
        # name: (release table, reference view of real train, real holdout in the same view)
        "minimised raw": (train, train, holdout),
        f"k-anonymised (k={k}, l={l})": (ka.data, train_view, holdout_view),
        "twin of k-anon (GaussianCopula)": (twins["twin_of_k_anon_gc"], train_view, holdout_view),
        "twin of raw (GaussianCopula)": (twins["twin_of_raw_gc"], train, holdout),
        "twin of raw (CTGAN)": (twins["twin_of_raw_ctgan"], train, holdout),
    }
    rows = []
    dcr_dists = {}
    for name, (rel, ref, hold) in releases.items():
        is_twin = name.startswith("twin")
        row = {"release": name, "rows": len(rel)}
        if not is_twin:
            r = reid_risk(rel, qis)
            row.update({"unique_share": r["unique_share"], "avg_reid_risk": r["avg_risk"], "max_reid_risk": r["max_risk"]})
        la = E.linkage_attack(train, ref, rel, qis, SENSITIVE)
        row.update({
            "linkage_matched": la["matched_share"],
            "belief_if_true": la["belief_for_true_positives"],
            "prior": la["prior"],
            "certain_disclosures": la["certain_disclosures"],
        })
        if is_twin:
            d, d_h, d_s = E.dcr(ref, hold, rel)
            row.update({"exact_copies": d["exact_copies_synthetic"], "exact_copies_holdout": d["exact_copies_holdout"],
                        "dcr_median": d["dcr_synthetic_median"], "dcr_holdout_median": d["dcr_holdout_median"]})
            dcr_dists[name] = (d_h, d_s)
        sim = E.marginal_similarity(ref, rel)
        row.update({
            "marginal_similarity": float(np.mean(list(sim.values()))),
            "association_gap": E.association_gap(ref, rel),
            "ml_auc_at_risk": E.ml_utility(rel, hold, TARGET, seed),
        })
        rows.append(row)
    results = pd.DataFrame(rows)

    # A concrete analyst question, answered on each release.
    def grade_by_prog(df: pd.DataFrame) -> pd.Series:
        return df.groupby("programme")["avg_grade"].mean().round(2)

    question = pd.DataFrame({"real (minimised)": grade_by_prog(train)})
    for name in ["twin of raw (GaussianCopula)", "twin of raw (CTGAN)"]:
        question[name] = grade_by_prog(releases[name][0])
    question_lvl = pd.DataFrame({
        "real": train_view.groupby("programme")["avg_grade"].mean().round(2),
        "k-anonymised": ka.data.groupby("programme")["avg_grade"].mean().round(2),
        "twin of k-anon": twins["twin_of_k_anon_gc"].groupby("programme")["avg_grade"].mean().round(2),
    })

    metrics = {
        "config": {"n_students": n_students, "n_tickets": n_tickets, "seed": seed, "k": k, "l": l,
                   "max_suppression": max_suppression, "ctgan_epochs": ctgan_epochs, "train_rows": len(train),
                   "holdout_rows": len(holdout)},
        "uniqueness_by_qi_set": uniq,
        "k_anonymity": {"node": ka.node, "suppressed": ka.suppressed, "loss": ka.loss, "classes": ka.classes,
                        "nodes_checked": ka.nodes_checked, "nodes_feasible": ka.nodes_feasible},
        "text": text_summary,
        "text_by_label": text_table.reset_index().to_dict(orient="records"),
        "releases": results.to_dict(orient="records"),
        "timings_s": timings,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("6/6 writing report", flush=True)
    write_report(out, metrics, results, text_table, misses, pseudo, question, question_lvl, dcr_dists, students, ka, hier)
    return metrics
