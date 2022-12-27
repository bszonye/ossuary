"""Unit tests for bones.pmf.DicePMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

import bones.pmf
from bones.pmf import D, die_range, PMF


class TestDicePMFInit:
    """Test the PMF constructor."""

    # The module should provide prebuilt dice objects for all of these.
    die_sizes = (2, 3, 4, 6, 8, 10, 12, 20, 30, 100, 1000)

    @pytest.mark.parametrize("size", die_sizes)
    def test_DX_objects(self, size: int) -> None:
        """Test the predefined dice objects."""
        # For each predefined die size, get its PMF from the D function.
        die = D(size)
        assert isinstance(die, PMF)
        # Verify that there's a corresponding module variable. The
        # D function caches results, so they should be "is" equivalent.
        attr = f"D{size}"
        mdie = getattr(bones.pmf, attr)
        assert isinstance(mdie, PMF)
        assert die.mapping is mdie.mapping
        # Test the PMF properties.
        assert len(die) == size
        for v, p in die.mapping.items():
            assert type(v) is int
            assert type(p) is int
            assert p == 1
        assert die.total == size
        assert die.support == tuple(die_range(size))

    def test_special_dice(self) -> None:
        """Test special dice objects."""
        from bones.pmf import D00, D000, DF

        # Construct the special die ranges.
        r00 = die_range(0, 99)
        r000 = die_range(0, 999)
        rF = die_range(-1, +1)
        # Verify face values for special dice.
        assert D00.support == tuple(r00)
        assert D000.support == tuple(r000)
        assert DF.support == tuple(rF) == (-1, 0, +1)
        # Verify "is" equivalence to the D function for special dice.
        assert D00.mapping is D(r00).mapping
        assert D000.mapping is D(r000).mapping
        assert DF.mapping is D(rF).mapping
