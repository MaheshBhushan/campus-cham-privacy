"""End-to-end run: generate -> minimise -> mask text -> k-anonymise -> synthesise -> attack -> report."""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as C
from . import evaluate as E
from .attacks import gdpr_criteria, membership_inference
from .dp import dp_synthesise, epsilon_sweep
from .generate import DIRECT_IDENTIFIERS, QUASI_IDENTIFIERS, generate_students, generate_tickets, staff_roster
from .indirect import anonymity_set_sizes, generalise_indirect
from .kanon import _ident, _star, anonymise, build_hierarchies, generalise, mondrian, ncp, reid_risk
from .report import write_report
from .synth import synthesise
from .textmask import anonymise_tickets

TARGET = "at_risk"                # analysis target for the ML-utility test
SENSITIVE = "disability_support"  # GDPR Art. 9 attribute: l-diversity protects it, attacks target it
EXTRA_NUMERIC = ("semester", "ects", "avg_grade")  # released unchanged unless "all attributes" is chosen


def _timer(timings: dict[str, float]):
    def timed(name, fn, *a, **kw):
        t = time.perf_counter()
        r = fn(*a, **kw)
        timings[name] = round(time.perf_counter() - t, 2)
        print(f"  {name}: {timings[name]} s", flush=True)
        return r

    return timed


def _text(tickets: pd.DataFrame, students: pd.DataFrame, seed: int, gliner: bool, key: bytes, timed) -> dict:
    roster = [*students.first_name, *students.last_name, *staff_roster(seed)]
    variants = {"spaCy": dict(ner="spacy"), "spaCy + register": dict(ner="spacy", roster=roster)}
    if gliner:
        variants["GLiNER + register"] = dict(ner="gliner", roster=roster)
    scored = {}
    for name, kw in variants.items():
        masked = timed(f"text: {name}", anonymise_tickets, tickets, "mask", **kw)
        scored[name] = (masked, *E.text_scores(masked))
    main = "spaCy + register"
    masked, table, summary, misses = scored[main]
    masked = masked.copy()
    masked["text_final"] = [generalise_indirect(t, lang) for t, lang in zip(masked.text_anonymised, masked.lang)]
    before = anonymity_set_sizes(list(masked.text_anonymised), students)
    after = anonymity_set_sizes(list(masked.text_final), students)
    has = [b is not None for b in before]
    b = np.array([x for x, h in zip(before, has) if h], dtype=float)
    # "No self-description left" means the writer hides among all students.
    a = np.array([len(students) if x is None else x for x, h in zip(after, has) if h], dtype=float)
    pseudo = anonymise_tickets(tickets.head(8), "pseudonym", key, roster=roster)
    return {
        "masked": masked, "table": table, "summary": summary, "misses": misses, "pseudo": pseudo, "main": main,
        "variants": pd.DataFrame([
            {"pipeline": n, **{k: s[k] for k in ("recall_direct_identifiers", "leak_rate_direct_identifiers",
                                                  "indirect_identifiers_mean_coverage", "precision")},
             "person_leak_rate": float(t.loc["PERSON", "leak_rate"])}
            for n, (_, t, s, _) in scored.items()
        ]),
        "indirect": {"texts": int(len(b)), "unique_before": float((b == 1).mean()), "unique_after": float((a == 1).mean()),
                     "median_before": float(np.median(b)), "median_after": float(np.median(a)),
                     "min_before": int(b.min()), "min_after": int(a.min())},
        "indirect_examples": masked.loc[[h and t1 != t2 for h, t1, t2 in zip(has, masked.text_anonymised, masked.text_final)],
                                        ["lang", "text_anonymised", "text_final"]].groupby("lang").head(2),
    }


def _evaluate_release(name, rel, ori, view, hold, train, seed, n_attacks) -> tuple[dict, dict, list[str], tuple | None]:
    """ori: real rows that went into `rel`; view: all training rows in `rel`'s representation; hold: holdout likewise."""
    is_twin = name.startswith(("twin", "DP"))  # synthetic rows are nobody: no prosecutor risk, DCR instead
    row = {"release": name, "rows": len(rel)}
    if not is_twin:
        r = reid_risk(rel, QUASI_IDENTIFIERS)
        row.update({"unique_share": r["unique_share"], "avg_reid_risk": r["avg_risk"]})
    la = E.linkage_attack(train, view, rel, QUASI_IDENTIFIERS, SENSITIVE)
    row.update({"belief_if_true": la["belief_for_true_positives"], "prior": la["prior"],
                "certain_disclosures": la["certain_disclosures"]})
    dists = None
    if is_twin:
        d, d_h, d_s = E.dcr(view, hold, rel)
        row.update({"exact_copies": d["exact_copies_synthetic"], "dcr_median": d["dcr_synthetic_median"],
                    "dcr_holdout_median": d["dcr_holdout_median"]})
        dists = (d_h, d_s)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crit = gdpr_criteria(ori[rel.columns], rel, hold[rel.columns], SENSITIVE, n_attacks, seed)
    notes = sorted({str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)})
    mia = membership_inference(view[rel.columns], hold[rel.columns], rel)
    row.update({
        "singling_out": crit.loc["singling_out_multivariate", "risk"],
        "linkability": crit.loc["linkability", "risk"],
        "inference": crit.loc["inference", "risk"],
        "mia_auc": mia["auc"],
        # Generalised tables are compared with themselves in their own representation, so distribution
        # distortion is only meaningful for synthetic rows; their loss is reported as NCP instead.
        "marginal_similarity": float(np.mean(list(E.marginal_similarity(view, rel).values()))) if is_twin else np.nan,
        "association_gap": E.association_gap(view, rel) if is_twin else np.nan,
        "ml_auc_at_risk": E.ml_utility(rel, hold, TARGET, seed),
    })
    return row, {"criteria": crit.reset_index(names="criterion").to_dict(orient="records"), "mia": mia}, notes, dists


def run(
    out: Path,
    n_students: int = C.CAMPUS_CHAM_STUDENTS,
    n_tickets: int = 600,
    seed: int = 42,
    k: int = 5,
    l: int = 2,
    max_suppression: float = 0.05,
    ctgan_epochs: int = 300,
    epsilons: tuple[float, ...] = (1.0, 5.0),
    quick: bool = False,
    n_attacks: int = 500,
    pseudonym_key: bytes = b"demo-key-kept-by-the-data-protection-officer",
) -> dict:
    data_dir, fig_dir = out / "data", out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    timed = _timer(timings)

    print("1/7 generating fake Campus Cham data", flush=True)
    students = timed("generate students", generate_students, n_students, seed)
    tickets = timed("generate tickets", generate_tickets, students, n_tickets, seed)
    students.to_csv(data_dir / "students_raw.csv", index=False)
    tickets.to_csv(data_dir / "tickets_raw.csv", index=False)

    # Data minimisation (GDPR Art. 5(1)(c)): analysts never need names, contact data or IDs.
    research = students.drop(columns=DIRECT_IDENTIFIERS + ["degree"])  # degree is implied by programme
    # Own RNG stream: reusing the generator's seed would make the split correlate with programme.
    is_train = np.random.default_rng([seed, 2]).random(len(research)) < 0.8
    train, holdout = research[is_train].reset_index(drop=True), research[~is_train].reset_index(drop=True)
    train.to_csv(data_dir / "students_minimised.csv", index=False)

    print("2/7 free text: Presidio (+ register, + GLiNER) and indirect identifiers", flush=True)
    text = _text(tickets, students, seed, not quick, pseudonym_key, timed)
    text["masked"].to_csv(data_dir / "tickets_anonymised.csv", index=False)
    text["misses"].to_csv(out / "text_misses.csv", index=False)

    print(f"3/7 k-anonymity (k={k}, l={l} on {SENSITIVE})", flush=True)
    hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
    ka = timed("full-domain k-anonymity", anonymise, train, hier, k=k, max_suppression=max_suppression, sensitive=SENSITIVE, l=l)
    mo = timed("Mondrian k-anonymity", mondrian, train, hier, k=k, l=l, sensitive=SENSITIVE)
    # Variant that also coarsens grades/credits/semester: the singling-out test shows why it is needed.
    hier_all = {**hier, **{c: [_ident, _star] for c in EXTRA_NUMERIC}}
    numeric_all = ("age", "enrolment_year", *EXTRA_NUMERIC)
    mo_all = timed("Mondrian k-anonymity, all attributes", mondrian, train, hier_all, k=k, l=l, sensitive=SENSITIVE, numeric=numeric_all)
    train_view, holdout_view = generalise(train, hier, ka.node), generalise(holdout, hier, ka.node)
    for name, df in {"k_anon_full_domain": ka.data, "k_anon_mondrian": mo.data, "k_anon_mondrian_all": mo_all.data}.items():
        df.to_csv(data_dir / f"students_{name}.csv", index=False)
    def nat_kept(data: pd.DataFrame) -> float:
        """Share of training students whose nationality is released as a country or world region."""
        return float((~data.nationality.isin(["*", "International"])).sum() / len(train))

    # NCP of the "all attributes" variant also charges grades/credits/semester, so it is not on the same scale.
    kanon_cmp = pd.DataFrame([
        {"method": "full-domain (one level per attribute)", "loss_ncp": ncp(ka.data, train, hier),
         "groups": ka.classes, "suppressed": ka.suppressed, "nationality_kept": nat_kept(ka.data)},
        {"method": "Mondrian (per-group recoding)", "loss_ncp": mo.ncp, "groups": len(mo.partitions),
         "suppressed": 0, "nationality_kept": nat_kept(mo.data)},
        {"method": "Mondrian, grades/credits/semester too", "loss_ncp": mo_all.ncp, "groups": len(mo_all.partitions),
         "suppressed": 0, "nationality_kept": nat_kept(mo_all.data)},
    ])

    print("4/7 synthetic twins (SDV, smartnoise DP)", flush=True)
    twins = {
        "twin of k-anon (GaussianCopula)": timed("twin: GaussianCopula of k-anon", synthesise, ka.data, "gaussian_copula", len(train), seed),
        "twin (GaussianCopula)": timed("twin: GaussianCopula", synthesise, train, "gaussian_copula", len(train), seed),
        "twin (CTGAN)": timed("twin: CTGAN", synthesise, train, "ctgan", len(train), seed, ctgan_epochs),
    }
    for eps in epsilons:
        twins[f"DP twin (AIM, ε={eps:g})"] = timed(f"twin: DP AIM eps={eps:g}", dp_synthesise, train, eps, "aim", len(train), seed)
    for name, df in twins.items():
        slug = re.sub(r"[^a-z0-9.]+", "_", name.lower().replace("ε", "eps")).strip("_")
        df.to_csv(data_dir / f"students_{slug}.csv", index=False)

    print("5/7 attacking every release (singling out, linkability, inference, membership)", flush=True)
    mo_hold, mo_all_hold = mo.transform(holdout), mo_all.transform(holdout)
    releases = {
        # name: (release, real rows that went in, all train rows in release view, holdout in release view)
        "minimised raw": (train, train, train, holdout),
        f"k-anon full-domain (k={k}, l={l})": (ka.data, train_view.loc[ka.data.index], train_view, holdout_view),
        f"k-anon Mondrian (k={k}, l={l})": (mo.data, mo.data, mo.data, mo_hold),
        "k-anon Mondrian, all attributes": (mo_all.data, mo_all.data, mo_all.data, mo_all_hold),
        "twin of k-anon (GaussianCopula)": (twins["twin of k-anon (GaussianCopula)"], ka.data, train_view, holdout_view),
        **{n: (df, train, train, holdout) for n, df in twins.items() if n != "twin of k-anon (GaussianCopula)"},
    }
    rows, attack_detail, notes, dcr_dists = [], {}, {}, {}
    for name, (rel, ori, view, hold) in releases.items():
        row, detail, note, dists = timed(f"attack: {name}", _evaluate_release, name, rel, ori, view, hold, train, seed, n_attacks)
        rows.append(row)
        attack_detail[name] = detail
        if note:
            notes[name] = note
        if dists is not None:
            dcr_dists[name] = dists
    results = pd.DataFrame(rows)

    sweep = pd.DataFrame()
    if not quick:
        print("6/7 epsilon sweep (MST, AIM)", flush=True)
        sweep = pd.concat([timed(f"epsilon sweep {m}", epsilon_sweep, train, holdout, model=m, seed=seed) for m in ("mst", "aim")])
        sweep.to_csv(out / "epsilon_sweep.csv", index=False)

    # A concrete analyst question, answered on each release.
    question = pd.DataFrame({"real": train.groupby("programme")["avg_grade"].mean()})
    for name in ["twin (GaussianCopula)", "twin (CTGAN)", *[n for n in twins if n.startswith("DP")]]:
        question[name] = twins[name].groupby("programme")["avg_grade"].mean()
    question[f"k-anon Mondrian (k={k}, l={l})"] = mo.data.groupby("programme")["avg_grade"].mean()
    question = question.round(2)

    metrics = {
        "config": {"n_students": n_students, "n_tickets": n_tickets, "seed": seed, "k": k, "l": l,
                   "max_suppression": max_suppression, "ctgan_epochs": ctgan_epochs, "epsilons": list(epsilons),
                   "train_rows": len(train), "holdout_rows": len(holdout), "quick": quick, "n_attacks": n_attacks},
        "uniqueness_by_qi_set": {
            label: reid_risk(train, cols)["unique_share"] for label, cols in {
                "nationality + programme": ["nationality", "programme"],
                "+ gender": ["nationality", "programme", "gender"],
                "+ age": ["nationality", "programme", "gender", "age"],
                "all 6 QIs": QUASI_IDENTIFIERS,
            }.items()
        },
        "text": text["summary"],
        "text_variants": text["variants"].to_dict(orient="records"),
        "text_by_label": text["table"].reset_index().to_dict(orient="records"),
        "indirect": text["indirect"],
        "k_anonymity": {"node": ka.node, "suppressed": ka.suppressed, "loss": ka.loss, "classes": ka.classes,
                        "nodes_checked": ka.nodes_checked, "nodes_feasible": ka.nodes_feasible},
        "kanon_comparison": kanon_cmp.to_dict(orient="records"),
        "releases": results.to_dict(orient="records"),
        "attacks": attack_detail,
        "attack_notes": notes,
        "epsilon_sweep": sweep.to_dict(orient="records"),
        "timings_s": timings,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float, ensure_ascii=False))

    print("7/7 writing report", flush=True)
    write_report(out, metrics, results, text, kanon_cmp, mo, question, dcr_dists, sweep)
    return metrics
