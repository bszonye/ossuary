"""Unit tests for bones.pmf.DicePMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

import bones.pmf
from bones.pmf import D, DicePMF


class TestDicePMFInit:
    """Test the PMF constructor."""

    def test_pmf_simple(self) -> None:
        """Test with default arguments."""
        pmf = DicePMF()
        assert len(pmf) == 0

    def test_pmf_int(self) -> None:
        """Test with default arguments."""
        items = (0,)
        pmf = DicePMF(items)
        assert len(pmf) == 1
        assert pmf[0] == 1

    def test_pmf_copy(self) -> None:
        """Test copying another DicePMF."""
        items = (0,)
        pmf1 = DicePMF(items)
        pmf2 = DicePMF(pmf1)
        assert pmf1.pairs is pmf2.pairs
        assert pmf1.total is pmf2.total

    # The module should provide prebuilt dice objects for all of these.
    die_sizes = (2, 3, 4, 6, 8, 10, 12, 20, 30, 100, 1000)

    @pytest.mark.parametrize("size", die_sizes)
    def test_DX_objects(self, size: int) -> None:
        """Test the predefined dice objects."""
        # For each predefined die size, get its PMF from the D function.
        die = D(size)
        assert isinstance(die, DicePMF)
        # Verify that there's a corresponding module variable. The
        # D function caches results, so they should be "is" equivalent.
        attr = f"D{size}"
        mdie = getattr(bones.pmf, attr)
        assert isinstance(mdie, DicePMF)
        assert die.pairs is mdie.pairs
        # Test the PMF properties.
        assert len(die) == size
        for v, p in die.items():
            assert type(v) is int
            assert type(p) is int
            assert p == 1
        assert die.total == size
        assert die.support == tuple(range(1, size + 1))
