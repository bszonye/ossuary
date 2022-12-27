"""Unit tests for bones.pmf.PMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import math
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, cast

import pytest

from bones.pmf import PMF, WT


class TestPMFInit:
    """Test the PMF constructor."""

    def test_pmf_default(self) -> None:
        """Test with default arguments."""
        pmf = PMF()
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 1

    def test_pmf_copy(self) -> None:
        """Test copying another PMF."""
        items = {0: 1}
        pmf1 = PMF(items)
        pmf2 = PMF(pmf1)
        assert pmf1.mapping is pmf2.mapping
        assert pmf1.total is pmf2.total

    def test_pmf_dict(self) -> None:
        """Test initialization from a dict."""
        items = {0: 1, 3: 2, 7: 1}
        pmf = PMF(items)
        assert len(pmf) == len(items)
        assert pmf.mapping == items

    def test_pmf_counter(self) -> None:
        """Test initialization from a Counter."""
        items = (1, 1, 1, 2, 2, 3)
        counter = Counter(items)
        pmf = PMF(counter)
        assert len(pmf) == len(counter)
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}

    def test_pmf_iterable(self) -> None:
        """Test initialization from a Counter."""
        items = ((1, 3), (2, 2), (3, 1))
        pmf = PMF(items)
        assert len(pmf) == len(items)
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}
        assert pmf.weight_graph == items

    def test_pmf_pzero(self) -> None:
        """Test a PMF with a zero probability."""
        items = {1: 3, 2: 0, 3: 1}
        pmf = PMF(items)
        assert len(pmf) == len(items)
        assert pmf.mapping == items
        assert pmf.support == (1, 3)

    def test_pmf_empty(self) -> None:
        """Test normalize parameter on empty PMFs."""
        pmf = PMF()
        assert len(pmf) == 0
        assert pmf.total == 1

    type_errors: Any = (
        0,  # not iterable
        "nope",  # not a pair
        (1, 2, 3),  # not a pair
        {1: "foo", 2: "bar", 3: "baz"},  # not a probability
        (([], 1), ({}, 2)),  # not hashable
    )

    @pytest.mark.parametrize("error", type_errors)
    def test_pmf_type_error(self, error: Any) -> None:
        """Test bad inputs to the PMF constructor."""
        with pytest.raises(TypeError):
            PMF(error)


weights = {
    # Zero weights.
    (): 1,  # No weights.
    (0,): 1,  # No non-zero weights.
    (0, 0): 1,  # Multiple zero weights.
    # Integral weights with gcd == 1.
    (1,): 1,
    (1, 2): 3,
    (2, 3, 4): 9,
    # Integral weights with gcd > 1.
    (2, 4, 6): 6,
    (10, 15, 20): 9,
    (6, 12, 6, 12, 24): 10,
    # Fractional weights.
    (Fraction(1, 2),): 1,
    (Fraction(1, 3), Fraction(2, 3)): 3,
    (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)): 6,
    (Fraction(2, 5), Fraction(3, 5), Fraction(4, 5)): 9,
    # Mixed weights.
    (Fraction(5, 2), 3, Fraction(2, 3)): 37,
}


class TestPMFIntWeight:
    """Test the PMF.int_weight method."""

    @pytest.mark.parametrize("weights, int_weight", weights.items())
    def test_pmf_int_weight(self, weights: Sequence[WT], int_weight: int) -> None:
        """Test various weight distributions."""
        items = {i: weights[i] for i in range(len(weights))}
        pmf = PMF(items)
        assert pmf.int_weight == int_weight


class TestPMFNormalized:
    """Test the PMF.normalized method."""

    @pytest.mark.parametrize("weights, int_weight", weights.items())
    def test_normalized_default(self, weights: Sequence[WT], int_weight: int) -> None:
        """Test the default parameters with various item weights."""
        items = {i: weights[i] for i in range(len(weights))}
        pmf = PMF(items)
        npmf = pmf.normalized()
        assert len(npmf) == len(pmf)
        assert npmf.total == int_weight

        # All weights should be integers after default normalization.
        weights = tuple(npmf.values())
        for w in weights:
            assert isinstance(w, int)

        total = sum(weights)
        if total:
            assert npmf.total == total
            assert math.gcd(*cast(tuple[int, ...], weights)) == 1
        else:
            assert len(npmf.support) == 0

    def test_normalized_1(self) -> None:
        """Test a PMF normalized to 1."""
        items = {1: 4, 2: 3, 3: 2, 4: 1}
        pmf = PMF(items)
        assert pmf.mapping == {1: 4, 2: 3, 3: 2, 4: 1}

        npmf = pmf.normalized(1)
        assert npmf.total == 1
        assert npmf.mapping == {
            1: Fraction(2, 5),
            2: Fraction(3, 10),
            3: Fraction(1, 5),
            4: Fraction(1, 10),
        }
        for p in npmf.values():  # all fractions!
            assert isinstance(p, Fraction)

    def test_normalized_100(self) -> None:
        """Test a PMF normalized to 100."""
        items = {1: 4, 2: 3, 3: 2, 4: 1}
        pmf = PMF(items)
        assert pmf.mapping == {1: 4, 2: 3, 3: 2, 4: 1}

        npmf = pmf.normalized(100)
        assert npmf.mapping == {1: 40, 2: 30, 3: 20, 4: 10}
        for p in npmf.values():  # no fractions!
            assert isinstance(p, int)

    def test_normalized_error(self) -> None:
        """Test bad normalization parameters."""
        pmf = PMF()
        with pytest.raises(ValueError):
            pmf.normalized(-1)
