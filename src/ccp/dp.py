"""Differentially private synthetic twin with smartnoise-synth (MST and AIM).

What epsilon means. Differential privacy (DP) promises that the release looks almost the same
whether or not any one student is in the training table (more precisely: whether that student's
row is the real one or replaced by any other possible row). Epsilon bounds "almost": every possible
release becomes at most e^epsilon times more (or less) likely because of one student. With
epsilon = 1 that factor is e ≈ 2.7. Concretely, if an attacker believes beforehand that Anna gets
disability support with 16 % (odds 0.19), then whatever synthetic table we publish, the attacker's
odds may grow at most 2.7-fold, to 0.52, i.e. 34 %, no matter what else they know. Linking Anna by
her quasi-identifiers cannot beat this bound, unlike for k-anonymity. Smaller epsilon means more
noise: epsilon = 0.5 allows 1.65x, epsilon = 10 allows 22,000x (a weak promise). We use (epsilon,
delta)-DP with delta = 1e-9: the bound may fail with probability one in a billion. The number of rows
is treated as public.

How it works. Both models are "select, measure, generate" methods over discrete tables: they add
Gaussian noise to counts of single columns and pairs of columns (marginals), fit a graphical model
(Private-PGM) consistent with the noisy counts, and sample new rows from it. Everything after the
noisy counts is post-processing, which cannot weaken the guarantee.

Budget split. The (epsilon, delta) pair is converted to one zero-concentrated-DP budget rho.
- MST: rho / 3 for noisy one-way counts of all columns, rho / 3 for choosing which column pairs to
  keep (exponential mechanism building a maximum spanning tree), rho / 3 for noisy counts of the
  chosen pairs.
- AIM: starts with noisy one-way counts, then repeats rounds that spend 10 % on picking the worst-fit
  pair (exponential mechanism) and 90 % on measuring it, until rho is used up.
- Preprocessing: 0. Columns are discretised with PUBLIC bins and categorical domains come from
  ccp.calibration, never from the data (data-derived min/max or category lists would leak, e.g. the
  one Ecuadorian student). Out-of-range numbers are clipped to the public bounds; categories outside
  the public domain raise an error instead of being learnt.

Decoding. A bin becomes a number by drawing uniformly among the public grid values inside it (e.g.
the ECTS bin 10-19 gives 10 or 15, the grade bin "1.3" gives 1.2, 1.3 or 1.4), so the output has the
input's columns and dtypes.

Reproducibility. smartnoise draws its Gaussian noise from OpenDP's secure sampler, which cannot be
seeded. For reproducible experiments we replace it with a seeded numpy Gaussian of the same scale.
The guarantee's maths is unchanged, but numpy floating-point noise is not hardened against
floating-point side-channel attacks; a production release should keep OpenDP's sampler.
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import snsynth.aim.aim as _aim
import snsynth.mst.mst as _mst
from snsynth.aim import AIMSynthesizer
from snsynth.mst import MSTSynthesizer
from snsynth.transform import LabelTransformer, TableTransformer

from . import calibration as C
from . import evaluate as E
from .generate import GRADE_STEPS, QUASI_IDENTIFIERS

DELTA = 1e-9
MODELS = {"mst": MSTSynthesizer, "aim": AIMSynthesizer}


# Numeric columns: (public grid of possible values, interior bin edges). Bounds follow the
# calibration: ages 18-49, at most 10 semesters / 210 ECTS (7-semester Bachelor), enrolment in the
# last ten winter terms, grades 1.0-4.0 to one decimal binned on the official German grade steps.
NUMERIC: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "age": (np.arange(18, 50), np.arange(20, 50, 2) - 0.5),  # 2-year bins
    "semester": (np.arange(1, 11), np.arange(2, 11) - 0.5),  # exact
    "enrolment_year": (np.arange(C.CURRENT_YEAR - 10, C.CURRENT_YEAR + 1),
                       np.arange(C.CURRENT_YEAR - 9, C.CURRENT_YEAR + 1) - 0.5),  # exact
    "ects": (np.arange(0, 211, 5), np.arange(10, 211, 10) - 2.5),  # 10-credit bins
    "avg_grade": (np.round(np.arange(10, 41) / 10, 1), (GRADE_STEPS[:-1] + GRADE_STEPS[1:]) / 2),
}
CATEGORICAL = {
    "postcode": sorted({*C.POSTCODES_DOMESTIC, *C.POSTCODES_INTERNATIONAL}),
    "gender": list(C.GENDER),
    "nationality": ["Germany", *C.ORIGIN_SOURCED, *C.ORIGIN_TAIL],
    "programme": list(C.PROGRAMMES),
    "disability_support": [False, True],
    "at_risk": [False, True],
}


def _cardinality(col: str) -> int:
    return len(NUMERIC[col][1]) + 1 if col in NUMERIC else len(CATEGORICAL[col])


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map every column to integer bin codes 0..cardinality-1 using public bins only."""
    unknown = set(df.columns) - set(NUMERIC) - set(CATEGORICAL)
    if unknown:
        raise ValueError(f"no public domain for columns {sorted(unknown)}")
    codes = {}
    for col in df.columns:
        if col in NUMERIC:
            grid, edges = NUMERIC[col]
            codes[col] = np.digitize(np.clip(df[col].to_numpy(dtype=float), grid[0], grid[-1]), edges)
        else:
            codes[col] = pd.Categorical(df[col], categories=CATEGORICAL[col]).codes
            if (codes[col] < 0).any():
                outside = sorted(set(df.loc[codes[col] < 0, col].astype(str)))
                raise ValueError(f"{col}: values outside the public domain {outside}")
    return pd.DataFrame(codes, index=df.index)


def _decode(codes: pd.DataFrame, like: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = {}
    for col in like.columns:
        c = codes[col].to_numpy(dtype=int)
        if col in NUMERIC:
            grid, edges = NUMERIC[col]
            grid_code = np.digitize(grid, edges)
            values = np.empty(len(c), dtype=grid.dtype)
            for b in np.unique(c):
                hit = c == b
                values[hit] = rng.choice(grid[grid_code == b], hit.sum())
            out[col] = values
        else:
            out[col] = np.asarray(CATEGORICAL[col], dtype=object)[c]
    return pd.DataFrame(out).astype(like.dtypes.to_dict())


def _public_transformer(columns: list[str]) -> TableTransformer:
    """Identity label transformer whose domain is every public code, fitted without the data."""
    cards = [_cardinality(c) for c in columns]
    tt = TableTransformer([LabelTransformer(nullable=False) for _ in columns])
    tt.fit(pd.DataFrame({c: np.arange(max(cards)) % k for c, k in zip(columns, cards)}))
    return tt


@contextlib.contextmanager
def _seeded_noise(rng: np.random.Generator) -> Iterator[None]:
    """Swap OpenDP's unseedable Gaussian sampler for a seeded numpy one (see module docstring)."""
    original = _mst.gaussian_noise, _aim.gaussian_noise

    def gaussian_noise(sigma: float, size: int | None = None) -> np.ndarray | float:
        return rng.normal(0.0, sigma, size)

    _mst.gaussian_noise = _aim.gaussian_noise = gaussian_noise
    try:
        yield
    finally:
        _mst.gaussian_noise, _aim.gaussian_noise = original


def dp_synthesise(df: pd.DataFrame, epsilon: float, model: str = "mst", n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """(epsilon, 1e-9)-DP synthetic copy of the raw research table (columns as in ccp.pipeline)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}, choose from {sorted(MODELS)}")
    codes = _encode(df)
    columns = list(codes.columns)
    # MST/AIM and Private-PGM draw selections and samples from numpy's global RNG.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    synth = MODELS[model](epsilon=float(epsilon), delta=DELTA)
    with _seeded_noise(rng), warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        synth.fit(codes, transformer=_public_transformer(columns), preprocessor_eps=0.0)
        # Not synth.sample(): it uses mbi's default "round" sampler, which in mbi 1.1 hands out the
        # values of every child column in the same row order within each parent group, so siblings
        # in the tree become strongly correlated (e.g. gender vs disability Cramér's V 0.04 -> 0.65
        # even at epsilon 1000). Plain i.i.d. sampling from the fitted model keeps its marginals.
        sample = synth.synthesizer.synthetic_data(rows=n or len(df), method="sample")
        if model == "mst":
            sample = synth.undo_compress_fn(sample)  # merged rare categories back to the public domain
    # The public transformer is the identity on codes, so the model's columns are the codes.
    codes = sample.df[[f"col{i}" for i in range(len(columns))]].set_axis(columns, axis=1)
    return _decode(codes, df, rng)


def epsilon_sweep(
    df: pd.DataFrame,
    holdout: pd.DataFrame,
    epsilons: Sequence[float] = (0.5, 1, 2, 5, 10),
    model: str = "mst",
    seed: int = 0,
) -> pd.DataFrame:
    """Privacy-utility trade-off: one DP twin of `df` per epsilon, scored like the other releases."""
    rows = []
    for eps in epsilons:
        t = time.perf_counter()
        syn = dp_synthesise(df, eps, model, len(df), seed)
        runtime = time.perf_counter() - t
        la = E.linkage_attack(df, df, syn, QUASI_IDENTIFIERS, "disability_support")
        rows.append({
            "epsilon": float(eps),
            "model": model,
            "marginal_similarity": float(np.mean(list(E.marginal_similarity(df, syn).values()))),
            "association_gap": E.association_gap(df, syn),
            "ml_auc_at_risk": E.ml_utility(syn, holdout, "at_risk", seed),
            "linkage_matched": la["matched_share"],
            "belief_if_true": la["belief_for_true_positives"],
            "prior": la["prior"],
            "certain_disclosures": la["certain_disclosures"],
            "runtime_s": round(runtime, 2),
        })
    return pd.DataFrame(rows)
