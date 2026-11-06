import numpy as np
import pytest

from bc_analysis.data import load_dataset
from bc_analysis.stats import class_balance, cohens_d, compare_groups, correlated_pairs


def test_cohens_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([0.0, 2.0, 4.0])
    # 两组标准差均为 2，均值差为 2 → d = 1
    assert cohens_d(a, b) == pytest.approx(1.0)


def test_cohens_d_zero_variance():
    assert cohens_d(np.ones(3), np.ones(3)) == 0.0


def test_compare_groups_ranks_by_effect_size():
    tests = compare_groups(load_dataset())
    assert len(tests) == 30
    d = tests["cohens_d"].abs().to_numpy()
    assert np.all(d[:-1] >= d[1:])
    assert tests.loc[0, "significant"]
    assert (tests["p_adj"] >= tests["p_value"] - 1e-12).all()


def test_class_balance_sums_to_one():
    bal = class_balance(load_dataset())
    assert bal["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_correlated_pairs_threshold():
    pairs = correlated_pairs(load_dataset(), threshold=0.9)
    assert (pairs["r"].abs() >= 0.9).all()
    assert len(pairs) > 0
