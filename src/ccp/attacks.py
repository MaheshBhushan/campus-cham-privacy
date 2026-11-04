"""Attack-based privacy tests for a released student table.

The Article 29 Working Party (Opinion 05/2014 on anonymisation techniques) calls data anonymous
only if an attacker can do none of three things with it:

- Singling out: write a description ("the only 41-year-old with postcode 93413") that matches
  exactly one real person. Univariate attacks use one rare value or an extreme (min/max) of one
  column; multivariate attacks combine three columns taken from a release record.
- Linkability: connect two records that belong to the same person. Here an attacker holds two
  separate lists with different attributes of the same students and uses the release as the
  bridge: if both halves of a person lead to the same release record, the lists are linked.
- Inference: deduce a hidden attribute from the ones the attacker knows. Here the attacker knows
  a student's quasi-identifiers and guesses `disability_support` (a GDPR Art. 9 health-related
  attribute) from the closest release record.

We measure all three with anonymeter (Giomi et al., PoPETs 2023). Each attack runs twice: against
the real people whose records went into the release (`ori`) and against real people who were
held out and never used (`control`). The control run measures what the attacker learns from
general patterns, which any useful release must allow. anonymeter's `risk` is the part of the
attack's success that only works on the people inside the release, normalised by the room left
for improvement: risk = (attack_rate - control_rate) / (1 - control_rate). 0 means the attack is
no better on members than on strangers; 1 means it succeeds on every member it could possibly
gain on. anonymeter clips the value and its 95 % interval to [0, 1], so a release that does
"worse than strangers" shows 0. `baseline_rate` is a random-guessing attacker and is a sanity
check: an attack that does not beat it says nothing about the data.

Membership inference asks a simpler question: was this person in the source table at all?
The attacker measures each candidate's distance to the closest release record; members should
sit closer than non-members. ROC AUC 0.5 means no leakage. Following Carlini et al. (S&P 2022)
we also report the true-positive rate at a 1 % and 10 % false-positive rate, which shows whether
a few people can be identified with confidence even when the average looks harmless.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import BufferingHandler

import numpy as np
import pandas as pd
import polars as pl
from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator
from anonymeter.evaluators.inference_evaluator import evaluate_inference_guesses
from anonymeter.evaluators.linkability_evaluator import LinkabilityIndexes
from anonymeter.evaluators.singling_out_evaluator import (
    multivariate_singling_out_queries,
    univariate_singling_out_queries,
)
from anonymeter.stats.confidence import EvaluationResults
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from .evaluate import _gower_matrix
from .generate import QUASI_IDENTIFIERS

# Two auxiliary sources a realistic attacker could hold about Campus Cham students, each covering
# half of the quasi-identifiers and sharing none of them, so they cannot be joined directly:
# - a registration-style list (residents' or sports-club register): age, gender, home postcode;
# - a course-style list (cohort photo page, international-office or project-team list): nationality,
#   programme, year of enrolment.
# Linking them would attach e.g. a home address area to a named course participant.
LINK_AUX = (["age", "gender", "postcode"], ["nationality", "programme", "enrolment_year"])
# Each half matches several students exactly (in the raw table "24, male, 94469" fits more than one
# person) and anonymeter breaks such ties independently per half, so with 1 neighbour it misses
# links an attacker would find by intersecting the two candidate lists. 5 candidates per half is the
# group size k-anonymity (k=5) promises; chance overlaps are absorbed by the control run.
LINK_NEIGHBOURS = 5
MULTIVARIATE_COLS = 3          # anonymeter's default size of a multivariate singling-out query
MAX_QUERY_ATTEMPTS = 100       # per requested query: bounds the search on well-protected releases
COLUMNS = ["risk", "ci_low", "ci_high", "attack_rate", "baseline_rate", "control_rate"]


@contextmanager
def _surface(criterion: str) -> Iterator[None]:
    """Collect anonymeter's logged warnings (e.g. fewer queries than requested) and Python warnings
    (e.g. attack no better than random guessing) and re-raise them tagged with the criterion, so the
    caller sees each caveat next to the number it qualifies instead of an untagged log line."""
    logger = logging.getLogger("anonymeter")
    handler = BufferingHandler(capacity=10_000)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        logger.removeHandler(handler)
    for msg in [r.getMessage() for r in handler.buffer] + [str(w.message) for w in caught]:
        warnings.warn(f"anonymeter {criterion}: {msg}", RuntimeWarning, stacklevel=3)


def _counts(ev: SinglingOutEvaluator | LinkabilityEvaluator | InferenceEvaluator) -> EvaluationResults:
    # results() sanity-checks the attack against anonymeter's own baseline, which _row replaces;
    # _row builds the final results with the reported baseline and so repeats that check on it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ev.results()


def _row(res: EvaluationResults, n_baseline: int, n_control: int) -> dict[str, float]:
    final = EvaluationResults(n_attacks=(res.n_attacks_ori, res.n_attacks_baseline, res.n_attacks_control),
                              n_success=res.n_success, n_baseline=n_baseline, n_control=n_control)
    risk = final.risk()
    return {"risk": risk.value, "ci_low": risk.ci[0], "ci_high": risk.ci[1], "attack_rate": final.attack_rate.value,
            "baseline_rate": final.baseline_rate.value, "control_rate": final.control_rate.value}


def _singled_out(df: pd.DataFrame, queries: list[pl.Expr]) -> int:
    """Number of queries that match exactly one distinct record of `df`."""
    counts = pl.DataFrame(df).unique().select([q.cast(pl.Int64).sum().alias(f"q{i}") for i, q in enumerate(queries)])
    return sum(c == 1 for c in counts.row(0))


def _singling_out(ori: pd.DataFrame, release: pd.DataFrame, control: pd.DataFrame, mode: str,
                  n_attacks: int, seed: int) -> dict[str, float]:
    # Text columns as categoricals: polars lists unique strings in a different order in every
    # process, which would make anonymeter's random-query baseline irreproducible.
    ori, release, control = (
        d.astype({c: "category" for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])})
        for d in (ori, release, control))
    # Generate the queries the evaluator will use, with its seed, to learn how many the release offers
    # (anonymeter divides by the requested number even if fewer exist) and to re-use them on control.
    # Univariate takes every rare value / extreme the release has: a fixed set of at most a few hundred.
    rng = np.random.default_rng(seed)
    syn_unique = pl.DataFrame(release).unique(maintain_order=True)
    if mode == "univariate":
        queries = univariate_singling_out_queries(syn_unique, syn_unique.height * syn_unique.width, rng)
    else:
        queries = multivariate_singling_out_queries(syn_unique, n_attacks, MULTIVARIATE_COLS,
                                                    MAX_QUERY_ATTEMPTS * n_attacks, rng)
    if not queries:
        # The generator found no value (univariate) or value combination (multivariate) held by a single
        # release row, so there is nothing to single out with: the attack fails outright.
        warnings.warn(f"no {mode} query isolates a single release record; reporting risk 0", RuntimeWarning)
        return {"risk": 0.0, "ci_low": 0.0, "ci_high": 0.0, "attack_rate": 0.0, "baseline_rate": float("nan"),
                "control_rate": float("nan")}
    # The control run is counted here, at the control set's own size, because anonymeter's correction
    # for a control set smaller than `ori` is unusable: version 1.1.0 fits the number of queries
    # instead of the number of successes, and below 1000 control rows it samples a single size, so
    # the factor depends on how many queries were asked (about 0.27 for 100 queries), not on the data.
    # A refit on real control subsamples cannot extrapolate from ~190 to ~720 rows either (checked
    # against fresh generator draws). Queries taken from a ~720-row release match roughly one person
    # per 720, and such a query singles out more often in a bigger crowd, so a smaller control set
    # under-counts control successes and the reported risk errs on the high side. On fresh non-members,
    # 500 multivariate queries singled out 77 people per 193 rows but 126 per 717 (raw release) and
    # 67 vs 96 (GaussianCopula twin), which overstates the twin's risk by about 0.05.
    ev = SinglingOutEvaluator(ori=ori, syn=release, n_attacks=len(queries), n_cols=MULTIVARIATE_COLS,
                              max_attempts=MAX_QUERY_ATTEMPTS * n_attacks, seed=seed)
    res = _counts(ev.evaluate(mode=mode))
    return _row(res, res.n_baseline, _singled_out(control, queries))


def _linkability(ori: pd.DataFrame, release: pd.DataFrame, control: pd.DataFrame, n_attacks: int,
                 seed: int) -> dict[str, float]:
    ev = LinkabilityEvaluator(ori=ori, syn=release, aux_cols=LINK_AUX, n_attacks=n_attacks,
                              n_neighbors=LINK_NEIGHBOURS, control=control)
    np.random.seed(seed)  # anonymeter draws its target records with pandas' global-RNG sampling
    res = _counts(ev.evaluate(n_jobs=-1))
    # anonymeter's random-guess baseline uses an unseeded generator; the same draw (LINK_NEIGHBOURS
    # random release records per half of each target), seeded, keeps reruns identical.
    rng = np.random.default_rng(seed)
    guess = [np.array([rng.choice(len(release), LINK_NEIGHBOURS, replace=False) for _ in range(n_attacks)])
             for _ in range(2)]
    return _row(res, LinkabilityIndexes(*guess).count_links(LINK_NEIGHBOURS), res.n_control)


def _inference(ori: pd.DataFrame, release: pd.DataFrame, control: pd.DataFrame, secret: str, n_attacks: int,
               seed: int) -> dict[str, float]:
    regression = pd.api.types.is_float_dtype(release[secret])
    ev = InferenceEvaluator(ori=ori, syn=release, aux_cols=QUASI_IDENTIFIERS, secret=secret, regression=regression,
                            n_attacks=n_attacks, control=control)
    np.random.seed(seed)  # anonymeter draws its target records with pandas' global-RNG sampling
    res = _counts(ev.evaluate(n_jobs=-1))
    # anonymeter 1.1.0's naive baseline aligns random release values to the targets by index label,
    # so most guesses become missing and count as wrong. Redo it positionally: each target's secret
    # is guessed as the secret of a random release record.
    rng = np.random.default_rng(seed)
    n = res.n_attacks_baseline
    secrets = ori[secret].iloc[rng.choice(len(ori), n, replace=False)].reset_index(drop=True)
    guesses = release[secret].iloc[rng.choice(len(release), n)].reset_index(drop=True)
    return _row(res, int(evaluate_inference_guesses(guesses, secrets, regression).sum()), res.n_control)


def gdpr_criteria(
    ori: pd.DataFrame,
    release: pd.DataFrame,
    control: pd.DataFrame,
    secret: str = "disability_support",
    n_attacks: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Singling out (univariate, multivariate), linkability and inference risk of `release`.

    `ori` are the real records that went into the release and `control` real records that did not;
    both must be in the release's representation (e.g. generalised to the same k-anonymity node).
    Linkability and inference attack the same number of people in `ori` and `control`, so their
    `n_attacks` is capped at the control size. Multivariate singling out asks `n_attacks` queries
    (fewer if the release yields fewer); univariate uses every query the release yields. Singling-out
    queries are checked on the whole control set; when it is smaller than `ori` this makes the
    singling-out risk err on the high side (see `_singling_out`).
    """
    if list(ori.columns) != list(release.columns) or list(control.columns) != list(release.columns):
        raise ValueError("ori, release and control must have the same columns in the same order")
    n_targets = min(n_attacks, len(control), len(ori))
    rows = {}
    for mode in ("univariate", "multivariate"):
        with _surface(f"singling out ({mode})"):
            rows[f"singling_out_{mode}"] = _singling_out(ori, release, control, mode, n_attacks, seed)
    with _surface("linkability"):
        rows["linkability"] = _linkability(ori, release, control, n_targets, seed)
    with _surface("inference"):
        rows["inference"] = _inference(ori, release, control, secret, n_targets, seed)
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)


def membership_inference(train: pd.DataFrame, holdout: pd.DataFrame, release: pd.DataFrame) -> dict:
    """Distance-to-closest-record membership attack: score = minus the Gower distance from a candidate
    to its nearest release record. `train` (members) and `holdout` (non-members) must be in the
    release's representation. Returns ROC AUC and TPR at 1 % / 10 % FPR."""
    r, t, h = _gower_matrix(release, train, holdout)
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(r)
    score = -np.concatenate([nn.kneighbors(t)[0].ravel(), nn.kneighbors(h)[0].ravel()])
    member = np.r_[np.ones(len(t), dtype=bool), np.zeros(len(h), dtype=bool)]
    # Every ROC corner is kept and the curve is read by linear interpolation: when members and
    # non-members tie at the same distance (common for generalised releases), the attacker can only
    # pick among the tied candidates at random, which is what the straight segment between corners
    # describes and what ROC AUC assumes too.
    fpr, tpr, _ = roc_curve(member, score, drop_intermediate=False)
    return {"auc": float(roc_auc_score(member, score)), "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
            "tpr_at_10pct_fpr": float(np.interp(0.10, fpr, tpr))}
