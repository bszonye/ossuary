"""Unit tests for bones.warhammer.AttackPMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from fractions import Fraction
from typing import Any

import pytest

from bones.pmf import Probability
from bones.warhammer import AttackCounter, AttackPMF


class TestAttackPMFInit:
    """Test the AttackPMF constructor."""

    def test_default(self) -> None:
        """Test with default arguments."""
        pmf = AttackPMF()
        assert len(pmf) == 0

    def test_attack_copy(self) -> None:
        """Test copying another AttackPMF."""
        pmf1 = AttackPMF()
        pmf2 = AttackPMF(pmf1)
        assert pmf1.pairs == pmf2.pairs
        assert pmf1.total_weight == pmf2.total_weight

    def test_attack_int(self) -> None:
        """Test with integer arguments."""
        pmf = AttackPMF({3: 1})
        assert len(pmf) == 1
        assert pmf.pairs == {AttackCounter(3): 1}

    def test_attack_iterable(self) -> None:
        """Test a PMF with more than one value."""
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(3)
        pmap: dict[AttackCounter, Probability] = {
            counter1: 1,
            counter2: Fraction(2),
            counter3: Fraction(3, 1),
        }
        plist = [(k, v) for k, v in pmap.items()]
        pmf = AttackPMF(plist)
        assert len(pmf) == 3
        assert pmf[counter1] == pmap[counter1]
        assert pmf[counter2] == pmap[counter2]
        assert pmf[counter3] == pmap[counter3]

    def test_attack_mapping(self) -> None:
        """Test a PMF with more than one value."""
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(2)
        pmap: dict[AttackCounter, Probability] = {
            counter1: 1,
            counter2: Fraction(2),
            counter3: Fraction(3, 1),
        }
        pmf = AttackPMF(pmap)
        assert len(pmf) == 2
        assert pmf[counter1] == pmap[counter1]
        assert pmf[counter2] == pmap[counter2]
        assert pmf[counter3] == pmap[counter3]

    def test_type_error(self) -> None:
        """Test parameter type errors."""
        items: Any
        with pytest.raises(TypeError):
            items = {"nope": 1}
            AttackPMF(items)
        with pytest.raises(TypeError):
            items = ("nope",)
            AttackPMF(items)
        with pytest.raises(TypeError):
            items = (("nope", 1),)
            AttackPMF(items)
