"""Unit tests for bones.warhammer.AttackPMF class."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from collections import Counter
from typing import Any

import pytest

from bones.pmf import WT
from bones.warhammer import AttackCounter, AttackPMF


class TestAttackCounterInit:
    """Test the AttackCounter post-init constructor."""

    def test_simple(self) -> None:
        """Test with minimal arguments."""
        counter = AttackCounter(0)
        assert counter.attacks == 0
        assert counter.wounds == 0
        assert counter.mortals == 0

    def test_fields(self) -> None:
        """Test with all parameters set."""
        counter = AttackCounter(3, 2, 1)
        assert counter.attacks == 3
        assert counter.wounds == 2
        assert counter.mortals == 1

    errors = [
        None,
        0.0,
        float("nan"),
        float("inf"),
        0 + 0j,
        "nope",
    ]

    @pytest.mark.parametrize("error", errors)
    def test_errors(self, error: Any) -> None:
        """Test with bad parameters."""
        with pytest.raises(TypeError):
            AttackCounter(attacks=error)
        with pytest.raises(TypeError):
            AttackCounter(0, wounds=error)
        with pytest.raises(TypeError):
            AttackCounter(0, mortals=error)


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
        assert pmf1.mapping == pmf2.mapping
        assert pmf1.total == pmf2.total

    def test_attack_int(self) -> None:
        """Test with integer arguments."""
        pmf = AttackPMF({3: 1})
        assert len(pmf) == 1
        assert pmf.mapping == {AttackCounter(3): 1}

    def test_attack_tuples(self) -> None:
        """Test with tuple arguments."""
        tuples = (
            (1,),
            (2, 1),
            (3, 2, 1),
        )
        pmf = AttackPMF(tuples)
        assert len(pmf) == len(tuples)
        for item in tuples:
            assert AttackCounter(*item) in pmf

    def test_attack_iterable(self) -> None:
        """Test a PMF with more than one value."""
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(3)
        pmap: dict[AttackCounter, WT] = {
            counter1: 3,
            counter2: 2,
            counter3: 1,
        }
        pmf = AttackPMF(Counter(pmap).elements())
        assert len(pmf) == 3
        assert pmf.weight(counter1) == pmap[counter1]
        assert pmf.weight(counter2) == pmap[counter2]
        assert pmf.weight(counter3) == pmap[counter3]

    def test_attack_mapping(self) -> None:
        """Test a PMF with more than one value."""
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(2)
        pmap: dict[AttackCounter, WT] = {
            counter1: 3,
            counter2: 2,
            counter3: 1,
        }
        pmf = AttackPMF(pmap)
        assert len(pmf) == 2
        assert pmf.weight(counter1) == pmap[counter1]
        assert pmf.weight(counter2) == pmap[counter2]
        assert pmf.weight(counter3) == pmap[counter3]

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
            items = "nope"
            AttackPMF(items)
