"""Unit tests for the bones.roll module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

import bones.roll
from bones.pmf import PMF
from bones.roll import d, d00, d000, d3, d6, d20, dF, Dice, Die


class TestDieInit:
    def test_default(self) -> None:
        die = Die()
        assert len(die) == 6
        assert die.total == 6
        assert die.mapping == d6.mapping

    def test_int(self) -> None:
        die = Die(20)
        assert len(die) == 20
        assert die.total == 20
        assert die.mapping == d20.mapping

    def test_int_reverse(self) -> None:
        die = Die(6, reverse=True)
        assert len(die) == 6
        assert die.total == 6
        assert die.mapping == d6.mapping
        assert die.domain == (6, 5, 4, 3, 2, 1)

    def test_integers(self) -> None:
        die = Die((1, 2, 3))
        assert len(die) == 3
        assert die.total == 3
        assert die.mapping == d3.mapping

    def test_strings(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        die = Die(("cat", "dog", "bird", "fish", "snake"))
        assert len(die) == 5
        assert die.total == 5
        assert die.domain == pets

    def test_copy(self) -> None:
        # Copy another PMF (without normalization).
        die = Die(d6)
        assert die.mapping is d6.mapping
        assert die.total is d6.total


class TestDieOutput:
    def test_repr(self) -> None:
        assert repr(Die(())) == "Die(())"
        assert repr(Die()) == "Die((1, 2, 3, 4, 5, 6))"
        assert repr(Die(3, reverse=True)) == "Die((3, 2, 1))"


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


class TestDiceNdX:
    def test_ndx(self) -> None:
        dice = Dice.NdX(3, d6)
        assert dice.mapping == d6.combination_weights(3)
        assert dice.sum().mapping == (3 @ d6).mapping

    def test_ndx_keep_high(self) -> None:
        die = Dice.NdX(3, d3, kh=1).sum()
        assert die.mapping == {1: 1, 2: 7, 3: 19}

    def test_ndx_keep_low(self) -> None:
        die = Dice.NdX(3, d3, kl=1).sum()
        assert die.mapping == {1: 19, 2: 7, 3: 1}

    def test_ndx_keep_mid(self) -> None:
        die = Dice.NdX(3, d3, km=1).sum()
        assert die.mapping == {1: 7, 2: 13, 3: 7}

    def test_ndx_drop_all(self) -> None:
        # Drop all the dice.
        dhempty = Dice.NdX(3, d6, dh=3)
        dlempty = Dice.NdX(3, d6, dl=3)
        assert len(dhempty) == len(dlempty) == 0
        # Also OK to drop more than rolled.
        dhempty = Dice.NdX(1, d20, dh=3)
        dlempty = Dice.NdX(1, d20, dl=3)
        assert len(dhempty) == len(dlempty) == 0

    def test_errors(self) -> None:
        # Negative selectors.
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, dh=-1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, dl=-1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, kh=-1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, kl=-1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, km=-1)
        # More than one keep selector.
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, kh=1, kl=1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, kh=1, km=1)
        with pytest.raises(ValueError):
            Dice.NdX(1, d6, kl=1, km=1)


class TestDiceSum:
    def test_sum_int(self) -> None:
        combos = [(1, 2, 3), (4, 5, 6)]
        dice = Dice(combos)
        assert dice.sum().domain == (6, 15)

    def test_sum_float(self) -> None:
        combos = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        dice = Dice(combos)
        assert dice.sum().domain == (6.0, 15.0)

    def test_sum_str(self) -> None:
        combos = [("goat", "dog", "cat"), ("bird", "snake", "fish")]
        dice = Dice(combos)
        assert dice.sum().domain == ("goatdogcat", "birdsnakefish")
