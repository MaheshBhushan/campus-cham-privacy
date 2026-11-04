import pytest

from ccp import calibration as C
from ccp.generate import DIRECT_IDENTIFIERS, QUASI_IDENTIFIERS, generate_students
from ccp.kanon import anonymise, build_hierarchies, mondrian


@pytest.mark.parametrize("k,l,seed", [(2, 1, 1), (5, 2, 2), (10, 2, 3)])
def test_release_satisfies_k_and_l_within_suppression_budget(k, l, seed):
    df = generate_students(600, seed).drop(columns=DIRECT_IDENTIFIERS + ["degree"])
    hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
    res = anonymise(df, hier, k=k, max_suppression=0.05, sensitive="disability_support", l=l)

    groups = res.data.groupby(QUASI_IDENTIFIERS)
    assert groups.size().min() >= k
    assert groups["disability_support"].nunique().min() >= l
    assert res.suppressed == len(df) - len(res.data) <= 0.05 * len(df)


@pytest.mark.parametrize("k,l,seed", [(2, 1, 1), (5, 2, 2), (10, 2, 3)])
def test_mondrian_satisfies_k_and_l_without_suppression(k, l, seed):
    df = generate_students(600, seed).drop(columns=DIRECT_IDENTIFIERS + ["degree"])
    train, holdout = df.iloc[:500], df.iloc[500:]
    hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
    res = mondrian(train, hier, k=k, sensitive="disability_support", l=l)

    groups = res.data.groupby(QUASI_IDENTIFIERS)
    assert groups.size().min() >= k
    assert groups["disability_support"].nunique().min() >= l
    assert len(res.data) == len(train) and groups.ngroups == len(res.partitions)
    assert 0 < res.ncp < 1
    # Unseen records land in one of the released equivalence classes.
    released = set(map(tuple, res.data[QUASI_IDENTIFIERS].to_numpy()))
    assert all(tuple(r) in released for r in res.transform(holdout)[QUASI_IDENTIFIERS].to_numpy())


def test_impossible_budget_raises():
    df = generate_students(50, 0).drop(columns=DIRECT_IDENTIFIERS + ["degree"])
    hier = build_hierarchies(C.REGION, {p: v[1] for p, v in C.PROGRAMMES.items()})
    with pytest.raises(ValueError):
        anonymise(df, hier, k=60, max_suppression=0.0)
