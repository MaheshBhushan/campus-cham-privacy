"""k-anonymity + distinct l-diversity by full-domain generalisation (ARX-style lattice search).

Every quasi-identifier (QI) has a generalisation hierarchy: level 0 is the raw value,
the top level is "*". A lattice node picks one level per QI. For each node we generalise,
group records into equivalence classes (identical QI tuples), suppress records in classes
smaller than k (or with fewer than l distinct sensitive values), and accept the node if
suppression stays under a budget. Among accepted nodes we pick the lowest information loss.
The lattice here is small (a few hundred nodes), so exhaustive search is exact and fast.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Level = Callable[[pd.Series], pd.Series]


def _band(width: int) -> Level:
    def f(s: pd.Series) -> pd.Series:
        lo = (s.astype(int) // width) * width
        return lo.astype(str) + "-" + (lo + width - 1).astype(str)

    return f


def _star(s: pd.Series) -> pd.Series:
    return pd.Series("*", index=s.index)


def _ident(s: pd.Series) -> pd.Series:
    return s.astype(str)


def _prefix(n: int) -> Level:
    return lambda s: s.astype(str).str[:n] + "*" * (5 - n)


def _mapped(mapping: dict[str, str], default: str = "Other") -> Level:
    return lambda s: s.map(mapping).fillna(default)


def build_hierarchies(nationality_region: dict[str, str], programme_level: dict[str, str]) -> dict[str, list[Level]]:
    """Generalisation hierarchies for the Campus Cham student table."""
    region_coarse = {r: ("Germany" if r == "Germany" else "International") for r in set(nationality_region.values())}
    return {
        "age": [_ident, _band(5), _band(10), _star],
        "gender": [_ident, _star],
        "nationality": [
            _ident,
            _mapped(nationality_region),
            lambda s: s.map(nationality_region).map(region_coarse).fillna("International"),
            _star,
        ],
        "programme": [_ident, _mapped(programme_level), _star],
        "enrolment_year": [_ident, _band(2), _star],
        "postcode": [_ident, _prefix(3), _prefix(1), _star],
    }


def generalise(df: pd.DataFrame, hierarchies: dict[str, list[Level]], node: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    for col, lvl in node.items():
        out[col] = hierarchies[col][lvl](df[col])
    return out


@dataclass
class KAnonResult:
    node: dict[str, int]
    data: pd.DataFrame          # generalised, suppressed rows removed
    suppressed: int
    loss: float                 # mean normalised generalisation height, 0 = raw, 1 = all "*"
    classes: int
    nodes_checked: int
    nodes_feasible: int


def _violating_mask(codes: pd.DataFrame, k: int, sens: np.ndarray | None, l: int) -> np.ndarray:
    key = codes.groupby(list(codes.columns), sort=False).ngroup().to_numpy()
    size = np.bincount(key)[key]
    bad = size < k
    if sens is not None and l > 1:
        pairs = pd.DataFrame({"g": key, "s": sens}).drop_duplicates()
        distinct = np.bincount(pairs["g"].to_numpy(), minlength=key.max() + 1)
        bad |= distinct[key] < l
    return bad


def anonymise(
    df: pd.DataFrame,
    hierarchies: dict[str, list[Level]],
    k: int = 5,
    max_suppression: float = 0.05,
    sensitive: str | None = None,
    l: int = 1,
) -> KAnonResult:
    qis = list(hierarchies)
    heights = {q: len(hierarchies[q]) - 1 for q in qis}
    # Pre-compute every level of every QI once as integer codes; nodes only re-combine columns.
    values = {q: [h(df[q]) for h in hierarchies[q]] for q in qis}
    codes = {q: [pd.factorize(v)[0] for v in values[q]] for q in qis}
    sens = pd.factorize(df[sensitive])[0] if sensitive is not None else None
    best_node: dict[str, int] | None = None
    best_key: tuple[float, int] = (np.inf, 0)
    best_bad: np.ndarray | None = None
    checked = feasible = 0
    for combo in itertools.product(*(range(heights[q] + 1) for q in qis)):
        checked += 1
        node = dict(zip(qis, combo))
        bad = _violating_mask(pd.DataFrame({q: codes[q][node[q]] for q in qis}), k, sens, l)
        supp = int(bad.sum())
        if supp > max_suppression * len(df):
            continue
        feasible += 1
        loss = float(np.mean([node[q] / heights[q] for q in qis]))
        # Tie-break equal loss by fewer suppressed rows.
        if (loss, supp) < best_key:
            best_node, best_key, best_bad = node, (loss, supp), bad
    if best_node is None:
        raise ValueError(f"no generalisation reaches k={k}, l={l} within {max_suppression:.0%} suppression")
    gen = df.copy()
    for q in qis:
        gen[q] = values[q][best_node[q]].to_numpy()
    kept = gen.loc[~best_bad]
    return KAnonResult(best_node, kept, best_key[1], best_key[0], kept.groupby(qis).ngroups, checked, feasible)


def reid_risk(df: pd.DataFrame, qis: list[str]) -> dict[str, float]:
    """Prosecutor-model risk: an attacker who knows a person's QIs and that they are in the data."""
    size = df.groupby(qis, dropna=False)[qis[0]].transform("size").to_numpy()
    risk = 1.0 / size
    return {
        "records": len(df),
        "unique_share": float((size == 1).mean()),
        "share_risk_gt_20pct": float((risk > 0.2).mean()),
        "avg_risk": float(risk.mean()),
        "max_risk": float(risk.max()),
        "min_class_size": int(size.min()),
    }
