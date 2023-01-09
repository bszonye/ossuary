"""Unit tests for the bones.roll module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

import bones.roll
from bones.pmf import PMF
from bones.roll import d, d00, d000, d6, dF, Die


class TestDieInit:
    def test_ints(self) -> None:
        die = Die((1, 2, 3, 4, 5, 6))
        assert len(die) == 6
        assert die.total == 6
        assert die.mapping == d(6).mapping

    def test_copy(self) -> None:
        # Copy another PMF (without normalization).
        die = Die(d6)
        assert die.mapping is d6.mapping
        assert die.total is d6.total


class TestDieObjects:
    # The module should provide prebuilt dice objects for all of these.
    die_sizes = (2, 3, 4, 6, 8, 10, 12, 20, 30, 100, 1000)

    @pytest.mark.parametrize("size", die_sizes)
    def test_dX_objects(self, size: int) -> None:
        # For each predefined die size, get its PMF from the D function.
        die = d(size)
        assert isinstance(die, PMF)
        # Verify that there's a corresponding module variable.
        mdie: PMF[int] = getattr(bones.roll, f"d{size}")
        assert isinstance(mdie, PMF)
        assert die.mapping == mdie.mapping
        # Test the PMF properties.
        assert len(die) == size
        for v, p in die.mapping.items():
            assert type(v) is int
            assert type(p) is int
            assert p == 1
        assert die.total == size
        assert die.support == tuple(range(1, size + 1))

    def test_uncommon_dice(self) -> None:
        assert d00.support == tuple(range(100))
        assert d000.support == tuple(range(1000))
        assert dF.support == (-1, 0, +1)
