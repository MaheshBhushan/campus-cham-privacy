"""k-anonymity + distinct l-diversity: full-domain generalisation and Mondrian local recoding.

Every quasi-identifier (QI) has a generalisation hierarchy: level 0 is the raw value,
the top level is "*".

Full-domain (`anonymise`, ARX-style lattice search): a lattice node picks one level per QI
for the whole table. For each node we generalise, group records into equivalence classes
(identical QI tuples), suppress records in classes smaller than k (or with fewer than l
distinct sensitive values), and accept the node if suppression stays under a budget. Among
accepted nodes we pick the lowest (optionally weighted) information loss. The lattice here
is small (a few hundred nodes), so exhaustive search is exact and fast.

Local recoding (`mondrian`, top-down specialisation): start with one partition holding every
record at "*" and keep cutting partitions while every child still has >= k records and >= l
distinct sensitive values. Numeric QIs are cut at the median. Categorical QIs are specialised
one hierarchy level down: labels whose group passes k and l become children, the rest merge
into one residual child that keeps the parent label (e.g. "International" for small regions)
and must pass k and l itself. Each partition cuts the QI whose specialisation, followed as deep
as k and l allow on that QI alone, lowers the certainty penalty most (one-step gains undervalue
"Germany vs International", whose payoff is the regions below it). Dense regions (Germany,
South Asia, young students) keep fine detail while rare combinations stay coarse. No record is
suppressed.

Both are compared on the normalised certainty penalty (`ncp`): per record and QI, numeric
ranges cost their width over the attribute's range, categorical labels cost
(leaves covered - 1) / (domain size - 1); 0 = raw data, 1 = everything "*".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
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
    loss: float                 # (weighted) mean normalised generalisation height, 0 = raw, 1 = all "*"
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
    weights: dict[str, float] | None = None,
) -> KAnonResult:
    """Optimal full-domain node. `weights` (QI -> importance, missing QIs weigh 1) turns the loss
    into a weighted mean of normalised levels, so analysts can keep the QIs they care about."""
    qis = list(hierarchies)
    heights = {q: len(hierarchies[q]) - 1 for q in qis}
    if weights is not None and not set(weights) <= set(qis):
        raise ValueError(f"weights for unknown QIs: {sorted(set(weights) - set(qis))}")
    w = np.array([1.0 if weights is None else weights.get(q, 1.0) for q in qis])
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
        loss = float(np.average([node[q] / heights[q] for q in qis], weights=w))
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


def _level_labels(df: pd.DataFrame, hierarchies: dict[str, list[Level]], qis: list[str]) -> dict[str, list[np.ndarray]]:
    return {q: [h(df[q]).astype(str).to_numpy() for h in hierarchies[q]] for q in qis}


def _label_penalties(levels: list[np.ndarray]) -> list[dict[str, float]]:
    """Per hierarchy level: label -> (raw values it covers - 1) / (domain size - 1)."""
    raw = levels[0]
    denom = max(len(np.unique(raw)) - 1, 1)
    return [(pd.Series(raw).groupby(lab).nunique() - 1).div(denom).to_dict() for lab in levels]


def ncp(
    release: pd.DataFrame,
    df: pd.DataFrame,
    hierarchies: dict[str, list[Level]],
    numeric: tuple[str, ...] = ("age", "enrolment_year"),
) -> float:
    """Normalised certainty penalty of `release` (index subset of `df`) over all QIs and all rows of
    `df`; suppressed rows count as fully generalised. Numeric QIs accept raw values, "lo-hi" and "*"."""
    qis = list(hierarchies)
    pos = df.index.get_indexer(release.index)
    if (pos < 0).any():
        raise ValueError("release contains rows that are not in df")
    pen = np.ones((len(df), len(qis)))
    cats = [q for q in qis if q not in numeric]
    labels = _level_labels(df, hierarchies, cats)
    for j, q in enumerate(qis):
        lab = release[q].astype(str)
        if q in numeric:
            bounds = lab.str.extract(r"^(-?\d+(?:\.\d+)?)(?:-(-?\d+(?:\.\d+)?))?$").astype(float)
            star = (lab == "*").to_numpy()
            if (bounds[0].isna().to_numpy() & ~star).any():
                raise ValueError(f"unparsable {q} labels: {sorted(set(lab[bounds[0].isna() & ~star]))}")
            width = (bounds[1].fillna(bounds[0]) - bounds[0]).to_numpy()
            span = float(np.ptp(df[q].to_numpy(float)))
            p = np.where(star, 1.0, np.clip(width / span, 0.0, 1.0) if span else 0.0)
        else:
            # Most specific meaning wins when a label (e.g. "Germany") exists on several levels.
            mapping: dict[str, float] = {}
            for level in reversed(_label_penalties(labels[q])):
                mapping.update(level)
            mapped = lab.map(mapping)
            if mapped.isna().any():
                raise ValueError(f"unknown {q} labels: {sorted(set(lab[mapped.isna()]))}")
            p = mapped.to_numpy(float)
        pen[pos, j] = p
    return float(pen.mean())


@dataclass
class _Cut:
    qi: str
    value: float | None                 # numeric: first child takes values <= value
    level: int                          # categorical: children keyed by their label at this level
    keys: list[str]                     # categorical: labels of the specialised children
    children: list[_Cut | int]          # int = partition id; categorical residual child comes last
    fallback: int                       # child for labels not in keys: the residual, else the largest


@dataclass
class MondrianResult:
    data: pd.DataFrame                  # same rows and columns as the input, QIs as partition labels
    partitions: pd.DataFrame            # one row per partition: QI labels + size
    ncp: float
    hierarchies: dict[str, list[Level]] = field(repr=False)
    numeric: tuple[str, ...]
    tree: _Cut | int = field(repr=False)

    def transform(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Map unseen records into the release representation by following the cuts to the
        partition whose box contains them. Categorical values merged into a residual (or unseen in
        training) go to the residual child; without one, unseen categories and numeric values outside
        the training range follow the largest / outermost child."""
        qis = list(self.hierarchies)
        num = {q: new_df[q].to_numpy(float) for q in qis if q in self.numeric}
        labels = _level_labels(new_df, self.hierarchies, [q for q in qis if q not in self.numeric])
        pid = np.empty(len(new_df), dtype=int)
        stack: list[tuple[_Cut | int, np.ndarray]] = [(self.tree, np.arange(len(new_df)))]
        while stack:
            node, idx = stack.pop()
            if isinstance(node, int):
                pid[idx] = node
                continue
            if node.value is not None:
                child = (num[node.qi][idx] > node.value).astype(int)
            else:
                child = pd.Index(node.keys).get_indexer(labels[node.qi][node.level][idx])
                child[child < 0] = node.fallback
            stack += [(c, idx[child == i]) for i, c in enumerate(node.children)]
        out = new_df.copy()
        for q in qis:
            out[q] = self.partitions[q].to_numpy()[pid]
        return out


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def mondrian(
    df: pd.DataFrame,
    hierarchies: dict[str, list[Level]],
    k: int = 5,
    l: int = 1,
    sensitive: str | None = None,
    numeric: tuple[str, ...] = ("age", "enrolment_year"),
) -> MondrianResult:
    """Top-down local recoding to k-anonymity + distinct l-diversity without suppression."""
    qis = list(hierarchies)
    cats = [q for q in qis if q not in numeric]
    vals = {q: df[q].to_numpy(float) for q in qis if q in numeric}
    span = {q: float(np.ptp(v)) for q, v in vals.items()}
    labels = _level_labels(df, hierarchies, cats)
    penalty = {q: _label_penalties(labels[q]) for q in cats}
    sens = pd.factorize(df[sensitive])[0] if sensitive is not None and l > 1 else None

    width = sens.max() + 1 if sens is not None else 0

    def valid(child: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
        """Per child: >= k records and >= l distinct sensitive values."""
        ok = np.bincount(child, minlength=n) >= k
        if sens is not None:
            ok &= np.bincount(np.unique(child * width + sens[idx]) // width, minlength=n) >= l
        return ok

    Proposal = tuple[float, float | None, np.ndarray, list[str], bool]  # gain, cut, child, keys, residual

    def propose(q: str, idx: np.ndarray, level: dict[str, int]) -> Proposal | None:
        """The single allowed cut of `q` in this partition and its certainty-penalty gain."""
        if q in vals:
            v = vals[q][idx]
            lo, hi = v.min(), v.max()
            if lo == hi:
                return None
            cut = float(np.median(v))
            if cut >= hi:  # more than half the partition sits on the maximum
                cut = float(v[v < hi].max())
            child = (v > cut).astype(int)
            if not valid(child, idx, 2).all():
                return None
            left, right = v[child == 0], v[child == 1]
            gain = (len(v) * (hi - lo) - len(left) * np.ptp(left) - len(right) * np.ptp(right)) / span[q]
            return gain, cut, child, [], False
        lv = level[q]
        if lv == 0:
            return None
        # Partial specialisation: children that pass k and l get their own label, the rest
        # merge into one residual child that keeps the parent label and must pass itself.
        uniq, inv = np.unique(labels[q][lv - 1][idx], return_inverse=True)
        good = valid(inv, idx, len(uniq))
        rest = ~good[inv]
        residual = bool(rest.any())
        if not good.any() or (residual and not valid(np.zeros(rest.sum(), dtype=int), idx[rest], 1)[0]):
            return None
        keys = uniq[good].tolist()
        remap = np.full(len(uniq), len(keys))
        remap[good] = np.arange(len(keys))
        child = remap[inv]
        child_pen = np.array([penalty[q][lv - 1][key] for key in keys])
        parent_pen = penalty[q][lv][labels[q][lv][idx[0]]]
        gain = float(np.bincount(child, minlength=len(keys) + 1)[: len(keys)] @ (parent_pen - child_pen))
        return gain, None, child, keys, residual

    def reach(q: str, idx: np.ndarray, level: dict[str, int], p: Proposal | None) -> float:
        """Gain of specialising `q` alone as far as k and l allow, starting with proposal `p`.
        Ranking cuts by this lookahead instead of the one-step gain stops cheap first cuts
        (Germany vs International) from losing to numeric splits that later block the valuable
        ones (regions, countries)."""
        if p is None:
            return 0.0
        gain, cut, child, keys, _ = p
        sub = level if cut is not None else {**level, q: level[q] - 1}
        # The residual cannot be specialised on q again, so only named children recurse.
        return gain + sum(
            reach(q, idx[child == i], sub, propose(q, idx[child == i], sub)) for i in range(len(keys) or 2)
        )

    parts: list[tuple[np.ndarray, dict[str, int]]] = []

    def split(idx: np.ndarray, level: dict[str, int]) -> _Cut | int:
        best: tuple[tuple[float, float], str, Proposal] | None = None
        for q in qis:
            p = propose(q, idx, level)
            if p is None:
                continue
            score = (reach(q, idx, level, p), p[0])
            if best is None or score > best[0]:
                best = (score, q, p)
        if best is None:
            parts.append((idx, level))
            return len(parts) - 1
        _, q, (_, cut, child, keys, residual) = best
        if cut is None:
            sub = [{**level, q: level[q] - 1}] * len(keys) + [level] * residual
        else:
            sub = [level, level]
        children = [split(idx[child == i], s) for i, s in enumerate(sub)]
        fallback = len(keys) if residual else int(np.bincount(child).argmax())
        return _Cut(q, cut, level[q] - 1 if cut is None else 0, keys, children, fallback)

    root_level = {q: len(hierarchies[q]) - 1 for q in cats}
    root = np.arange(len(df))
    if not valid(np.zeros(len(df), dtype=int), root, 1)[0]:
        raise ValueError(f"the whole table does not reach k={k}, l={l}")
    tree = split(root, root_level)

    table = {q: [] for q in qis}
    pid = np.empty(len(df), dtype=int)
    for i, (idx, level) in enumerate(parts):
        pid[idx] = i
        for q in qis:
            if q in vals:
                lo, hi = vals[q][idx].min(), vals[q][idx].max()
                table[q].append(_fmt(lo) if lo == hi else f"{_fmt(lo)}-{_fmt(hi)}")
            else:
                table[q].append(labels[q][level[q]][idx[0]])
    partitions = pd.DataFrame(table).assign(size=[len(idx) for idx, _ in parts])
    out = df.copy()
    for q in qis:
        out[q] = partitions[q].to_numpy()[pid]
    return MondrianResult(out, partitions, ncp(out, df, hierarchies, numeric), hierarchies, tuple(numeric), tree)


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
