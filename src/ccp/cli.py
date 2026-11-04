"""ccp: Campus Cham confidential data pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import calibration as C


def main() -> None:
    p = argparse.ArgumentParser(prog="ccp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write fake students.csv and tickets.csv")
    g.add_argument("-o", "--out", type=Path, default=Path("data"))
    g.add_argument("-n", "--students", type=int, default=C.CAMPUS_CHAM_STUDENTS)
    g.add_argument("-t", "--tickets", type=int, default=600)
    g.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("run", help="full pipeline + attacks + report")
    r.add_argument("-o", "--out", type=Path, default=Path("results"))
    r.add_argument("-n", "--students", type=int, default=C.CAMPUS_CHAM_STUDENTS)
    r.add_argument("-t", "--tickets", type=int, default=600)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("-k", type=int, default=5)
    r.add_argument("-l", type=int, default=2, help="distinct l-diversity on disability_support (1 = off)")
    r.add_argument("--max-suppression", type=float, default=0.05)
    r.add_argument("--ctgan-epochs", type=int, default=300)
    r.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 5.0], help="ε of the DP (AIM) twins in the release table")
    r.add_argument("--quick", action="store_true", help="skip GLiNER and the ε sweep (~25 min faster)")

    b = sub.add_parser("bench", help="time generation, k-anonymity and synthesis on a large table")
    b.add_argument("-n", "--students", type=int, default=100_000)
    b.add_argument("--seed", type=int, default=42)

    lr = sub.add_parser("llm-rewrite", help="rewrite tickets with self-descriptions using a local LLM (slow on CPU)")
    lr.add_argument("--limit", type=int, default=16, help="number of tickets with a self-description to rewrite")
    lr.add_argument("-o", "--out", type=Path, default=Path("results/llm_rewrite.csv"))
    lr.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    if a.cmd == "generate":
        from .generate import write

        s, t = write(a.out, a.students, a.tickets, a.seed)
        print(f"wrote {len(s)} students and {len(t)} tickets to {a.out}/")
    elif a.cmd == "run":
        from .pipeline import run

        run(a.out, a.students, a.tickets, a.seed, a.k, a.l, a.max_suppression, a.ctgan_epochs, tuple(a.epsilons), a.quick)
        print(f"report: {a.out / 'report.md'}")
    elif a.cmd == "llm-rewrite":
        import numpy as np

        from .generate import generate_students, generate_tickets, staff_roster
        from .indirect import anonymity_set_sizes, generalise_indirect, has_indirect, llm_rewrite
        from .textmask import anonymise_tickets

        s = generate_students(C.CAMPUS_CHAM_STUDENTS, a.seed)
        roster = [*s.first_name, *s.last_name, *staff_roster(a.seed)]
        m = anonymise_tickets(generate_tickets(s, 600, a.seed), "mask", roster=roster)
        m = m[[has_indirect(t) for t in m.text_anonymised]].head(a.limit).copy()
        t0 = time.perf_counter()
        m["llm"] = llm_rewrite(list(m.text_anonymised), list(m.lang))
        m["rules"] = [generalise_indirect(t, lang) for t, lang in zip(m.text_anonymised, m.lang)]
        secs = time.perf_counter() - t0
        for col in ["text_anonymised", "rules", "llm"]:
            sizes = np.array([len(s) if x is None else x for x in anonymity_set_sizes(list(m[col]), s)], dtype=float)
            print(f"{col:16s} unique {np.mean(sizes == 1):.1%}  median anonymity set {np.median(sizes):.0f}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        m[["lang", "text_anonymised", "rules", "llm"]].to_csv(a.out, index=False)
        print(f"{len(m)} texts in {secs:.0f} s -> {a.out}")
    else:
        from .generate import DIRECT_IDENTIFIERS, QUASI_IDENTIFIERS, generate_students
        from .kanon import anonymise, build_hierarchies, mondrian, reid_risk
        from .synth import synthesise

        t0 = time.perf_counter()
        s = generate_students(a.students, a.seed)
        t1 = time.perf_counter()
        research = s.drop(columns=DIRECT_IDENTIFIERS + ["degree"])
        hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
        ka = anonymise(research, hier, k=5, max_suppression=0.05, sensitive="disability_support", l=2)
        t2 = time.perf_counter()
        mo = mondrian(research, hier, k=5, l=2, sensitive="disability_support")
        t3 = time.perf_counter()
        twin = synthesise(ka.data, "gaussian_copula", len(ka.data), a.seed)
        t4 = time.perf_counter()
        print(f"{a.students:,} students | generate {t1 - t0:.1f} s | full-domain k-anonymity ({ka.nodes_checked} nodes) "
              f"{t2 - t1:.1f} s | Mondrian {t3 - t2:.1f} s ({len(mo.partitions):,} groups, NCP {mo.ncp:.3f}) "
              f"| GaussianCopula twin {t4 - t3:.1f} s")
        print(f"unique on QIs before: {reid_risk(research, QUASI_IDENTIFIERS)['unique_share']:.1%}, "
              f"after: {reid_risk(ka.data, QUASI_IDENTIFIERS)['unique_share']:.1%}, suppressed {ka.suppressed}, "
              f"twin rows {len(twin):,}")


if __name__ == "__main__":
    main()
