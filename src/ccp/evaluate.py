"""Privacy and utility metrics for released tables and anonymised text."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# --- encoding helpers --------------------------------------------------------------------------


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _gower_matrix(ref: pd.DataFrame, *others: pd.DataFrame) -> list[np.ndarray]:
    """Encode so that L1 distance / n_columns equals the Gower distance."""
    num = _numeric_cols(ref)
    cat = [c for c in ref.columns if c not in num]
    lo, span = ref[num].min(), (ref[num].max() - ref[num].min()).replace(0, 1)
    frames = [ref, *others]
    onehot = pd.get_dummies(pd.concat([f[cat].astype(str) for f in frames], keys=range(len(frames))), dtype=float) * 0.5
    out = []
    for i, f in enumerate(frames):
        numeric = ((f[num] - lo) / span).to_numpy(dtype=float)
        out.append(np.hstack([numeric, onehot.xs(i).to_numpy()]) / len(ref.columns))
    return out


def dcr(train: pd.DataFrame, holdout: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Distance to closest record. Synthetic rows much closer to training rows than unseen real rows
    are (holdout) means the model memorised individuals. Returns summary, holdout and synthetic distances."""
    t, h, s = _gower_matrix(train, holdout, synthetic)
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(t)
    d_s = nn.kneighbors(s)[0].ravel()
    d_h = nn.kneighbors(h)[0].ravel()
    summary = {
        "dcr_synthetic_median": float(np.median(d_s)),
        "dcr_holdout_median": float(np.median(d_h)),
        "exact_copies_synthetic": float((d_s < 1e-9).mean()),
        "exact_copies_holdout": float((d_h < 1e-9).mean()),
        # Share of synthetic rows closer to train than the 5th percentile of holdout distances.
        "share_closer_than_holdout_p5": float((d_s < np.percentile(d_h, 5)).mean()),
    }
    return summary, d_h, d_s


def linkage_attack(real: pd.DataFrame, real_view: pd.DataFrame, release: pd.DataFrame, qis: list[str], target: str) -> dict[str, float]:
    """Attacker knows a person's quasi-identifiers (e.g. a classmate) and looks them up in the release
    to learn a sensitive yes/no attribute.

    Victims are the people who are unique on their raw QIs. `real_view` is the same people with QIs
    generalised to the release's level (identical to `real` for raw/synthetic-of-raw releases).
    The attacker's belief is the share of `target == True` among matching release rows. We report how
    often a victim is matched and, for victims who really have the attribute, the attacker's average
    belief compared with the prior (population base rate) — the gain is the disclosure.
    """
    sizes = real.groupby(qis, dropna=False)[qis[0]].transform("size")
    victims = real_view.loc[sizes.to_numpy() == 1, qis + [target]]
    belief = release.groupby(qis, dropna=False)[target].mean().rename("belief").reset_index()
    m = victims.merge(belief, on=qis, how="left")
    hit = m["belief"].notna()
    prior = float(real[target].mean())
    pos = m[target].astype(bool) & hit
    return {
        "unique_people": int(len(victims)),
        "matched_share": float(hit.mean()) if len(m) else 0.0,
        "prior": prior,
        "belief_for_true_positives": float(m.loc[pos, "belief"].mean()) if pos.any() else float("nan"),
        "certain_disclosures": int((m.loc[pos, "belief"] >= 0.999).sum()),
    }


# --- utility ------------------------------------------------------------------------------------


def _discretise(ref: pd.DataFrame, df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    out = df.copy()
    for c in _numeric_cols(ref):
        edges = np.unique(np.quantile(ref[c], np.linspace(0, 1, bins + 1)))
        out[c] = np.digitize(df[c], edges[1:-1]).astype(str) if len(edges) > 2 else "0"
    return out.astype(str)


def marginal_similarity(ref: pd.DataFrame, df: pd.DataFrame) -> dict[str, float]:
    """1 - total variation distance per column (1 = identical distribution)."""
    a, b = _discretise(ref, ref), _discretise(ref, df)
    out = {}
    for c in ref.columns:
        pa = a[c].value_counts(normalize=True)
        pb = b[c].value_counts(normalize=True)
        out[c] = float(1 - 0.5 * pa.subtract(pb, fill_value=0).abs().sum())
    return out


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct, correction=False)[0]
    return float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))


def association_gap(ref: pd.DataFrame, df: pd.DataFrame) -> float:
    """Mean absolute difference of pairwise Cramér's V (0 = relationships fully preserved)."""
    a, b = _discretise(ref, ref), _discretise(ref, df)
    cols = list(ref.columns)
    gaps = [
        abs(_cramers_v(a[c1], a[c2]) - _cramers_v(b[c1], b[c2]))
        for i, c1 in enumerate(cols) for c2 in cols[i + 1:]
    ]
    return float(np.mean(gaps))


def ml_utility(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int = 0) -> float:
    """Train on (possibly synthetic/anonymised) data, test on real unseen data: ROC AUC."""
    feats = [c for c in train.columns if c != target]
    both = pd.get_dummies(pd.concat([train[feats], test[feats]], keys=["tr", "te"]).astype(
        {c: str for c in feats if not pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])}
    ), dtype=float)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    clf.fit(both.xs("tr"), train[target].astype(bool))
    return float(roc_auc_score(test[target].astype(bool), clf.predict_proba(both.xs("te"))[:, 1]))


# --- text --------------------------------------------------------------------------------------


def text_scores(tickets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Per ground-truth span: recall (any overlap with a detection), coverage (share of the span's
    characters inside detections), leak (the original string still appears verbatim after
    anonymisation). Precision = detections that touch real personal data."""
    rows, misses = [], []
    n_det = n_det_true = 0
    for i, t in enumerate(tickets.itertuples()):
        gt = json.loads(t.spans)
        det = json.loads(t.detected)
        covered = np.zeros(len(t.text), dtype=bool)
        for d in det:
            covered[d["start"]:d["end"]] = True
        for g in gt:
            caught = any(d["start"] < g["end"] and g["start"] < d["end"] for d in det)
            coverage = float(covered[g["start"]:g["end"]].mean())
            # Whole-word match, so a short name like "Li" is not "found" inside "Lieferung".
            leaked = re.search(rf"(?<!\w){re.escape(g['text'])}(?!\w)", t.text_anonymised) is not None
            rows.append({"ticket": i, "label": g["label"], "lang": t.lang, "caught": caught,
                         "coverage": coverage, "leaked": leaked})
            if coverage < 1:
                misses.append({"label": g["label"], "lang": t.lang, "coverage": round(coverage, 2),
                               "pii_text": g["text"], "after": t.text_anonymised})
        for d in det:
            n_det += 1
            n_det_true += any(d["start"] < g["end"] and g["start"] < d["end"] for g in gt)
    per = pd.DataFrame(rows)
    table = per.groupby("label").agg(n=("caught", "size"), recall=("caught", "mean"),
                                     coverage=("coverage", "mean"), leak_rate=("leaked", "mean"))
    by_lang = per.groupby("lang").agg(recall=("caught", "mean"), leak_rate=("leaked", "mean"))
    direct = per[per.label != "INDIRECT_IDENTIFIER"]
    indirect = per[per.label == "INDIRECT_IDENTIFIER"]
    summary = {
        "spans": int(len(per)),
        "recall_direct_identifiers": float(direct.caught.mean()),
        "leak_rate_direct_identifiers": float(direct.leaked.mean()),
        "fully_masked_direct_identifiers": float((direct.coverage == 1).mean()),
        "indirect_identifiers_mean_coverage": float(indirect.coverage.mean()) if len(indirect) else float("nan"),
        "precision": n_det_true / n_det if n_det else float("nan"),
        "tickets_without_direct_leak": float(direct.groupby("ticket").leaked.sum().eq(0).mean()),
        **{f"recall_{k}": float(v) for k, v in by_lang.recall.items()},
        **{f"leak_rate_{k}": float(v) for k, v in by_lang.leak_rate.items()},
    }
    return table, summary, pd.DataFrame(misses)
