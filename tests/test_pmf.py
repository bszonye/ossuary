"""Unit tests for bones.pmf.PMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest

from bones.pmf import PMF, WT


class TestPMFInit:
    """Test the PMF constructor."""

    def test_pmf_default(self) -> None:
        """Test with default arguments."""
        pmf = PMF()
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 0

    def test_pmf_copy_normalized(self) -> None:
        """Test copying another PMF."""
        items = {0: 3}
        pmf1 = PMF(items, normalize=False)
        pmf2 = PMF(pmf1, normalize=True)
        assert pmf1.mapping is not pmf2.mapping
        assert pmf1.mapping == items
        assert pmf1.total == sum(items.values())
        assert pmf2.mapping == {0: 1}
        assert pmf2.total == 1

    def test_pmf_copy_exact(self) -> None:
        """Test copying another PMF."""
        items = {0: 3}
        pmf1 = PMF(items, normalize=False)
        pmf2 = PMF(pmf1, normalize=False)
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
        items = (1, 1, 1, 2, 2, 3)
        pmf = PMF(items)
        assert len(pmf) == 3
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}

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
        assert pmf.total == 0
        assert pmf.mapping == {}

    type_errors: Any = (
        0,  # not iterable
        {1: "foo", 2: "bar", 3: "baz"},  # not a probability
        ([], {}),  # not hashable
    )

    @pytest.mark.parametrize("error", type_errors)
    def test_pmf_type_error(self, error: Any) -> None:
        """Test bad inputs to the PMF constructor."""
        with pytest.raises(TypeError):
            PMF(error)

    def test_pmf_value_error(self) -> None:
        """Test bad inputs to the PMF constructor."""
        with pytest.raises(ValueError):
            PMF({0: -1})


weights = {
    # Zero weights.
    (): 0,  # No weights.
    (0,): 0,  # No non-zero weights.
    (0, 0): 0,  # Multiple zero weights.
    # Integral weights with gcd == 1.
    (1,): 1,
    (1, 2): 3,
    (2, 3, 4): 9,
    # Integral weights with gcd > 1.
    (2, 4, 6): 6,
    (10, 15, 20): 9,
    (6, 12, 6, 12, 24): 10,
}


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
        weights = tuple(npmf.weights)
        for w in weights:
            assert isinstance(w, int)

        total = sum(weights)
        if total:
            assert npmf.total == total
            assert math.gcd(*weights) == 1
        else:
            assert len(npmf.support) == 0
