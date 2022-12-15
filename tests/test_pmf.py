"""Unit tests for bones.pmf.PMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from collections.abc import Hashable

from bones.pmf import PMF, Probability


class TestPMFInit:
    """Test the PMF constructor."""

    def test_pmf_simple(self) -> None:
        """Test with default arguments."""
        pmf: PMF[Hashable] = PMF()
        assert len(pmf) == 0

    def test_pmf_int(self) -> None:
        """Test with default arguments."""
        items: dict[Hashable, Probability] = {0: 1}
        pmf: PMF[Hashable] = PMF(items)
        assert len(pmf) == 1
        assert pmf[0] == 1

    def test_pmf_copy(self) -> None:
        """Test copying another AttackPMF."""
        items: dict[Hashable, Probability] = {0: 1}
        pmf1: PMF[Hashable] = PMF(items)
        pmf2: PMF[Hashable] = PMF(pmf1)
        assert pmf1.pairs == pmf2.pairs
        assert pmf1.denominator == pmf2.denominator
