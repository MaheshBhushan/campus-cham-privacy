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

    r = sub.add_parser("run", help="full pipeline + evaluation + report")
    r.add_argument("-o", "--out", type=Path, default=Path("results"))
    r.add_argument("-n", "--students", type=int, default=C.CAMPUS_CHAM_STUDENTS)
    r.add_argument("-t", "--tickets", type=int, default=600)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("-k", type=int, default=5)
    r.add_argument("-l", type=int, default=2, help="distinct l-diversity on disability_support (1 = off)")
    r.add_argument("--max-suppression", type=float, default=0.05)
    r.add_argument("--ctgan-epochs", type=int, default=300)

    b = sub.add_parser("bench", help="time generation, k-anonymity and synthesis on a large table")
    b.add_argument("-n", "--students", type=int, default=100_000)
    b.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    if a.cmd == "generate":
        from .generate import write

        s, t = write(a.out, a.students, a.tickets, a.seed)
        print(f"wrote {len(s)} students and {len(t)} tickets to {a.out}/")
    elif a.cmd == "run":
        from .pipeline import run

        run(a.out, a.students, a.tickets, a.seed, a.k, a.l, a.max_suppression, a.ctgan_epochs)
        print(f"report: {a.out / 'report.md'}")
    else:
        from .generate import DIRECT_IDENTIFIERS, QUASI_IDENTIFIERS, generate_students
        from .kanon import anonymise, build_hierarchies, reid_risk
        from .synth import synthesise

        t0 = time.perf_counter()
        s = generate_students(a.students, a.seed)
        t1 = time.perf_counter()
        research = s.drop(columns=DIRECT_IDENTIFIERS + ["degree"])
        hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
        ka = anonymise(research, hier, k=5, max_suppression=0.05, sensitive="disability_support", l=2)
        t2 = time.perf_counter()
        twin = synthesise(ka.data, "gaussian_copula", len(ka.data), a.seed)
        t3 = time.perf_counter()
        print(f"{a.students:,} students | generate {t1 - t0:.1f} s | k-anonymity ({ka.nodes_checked} nodes) {t2 - t1:.1f} s "
              f"| GaussianCopula twin {t3 - t2:.1f} s")
        print(f"unique on QIs before: {reid_risk(research, QUASI_IDENTIFIERS)['unique_share']:.1%}, "
              f"after: {reid_risk(ka.data, QUASI_IDENTIFIERS)['unique_share']:.1%}, suppressed {ka.suppressed}, "
              f"twin rows {len(twin):,}")


if __name__ == "__main__":
    main()
