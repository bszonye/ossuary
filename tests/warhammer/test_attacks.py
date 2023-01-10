"""Unit tests for the AttackPMF & AttackCounter classes."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from collections import Counter

from bones.pmf import Weight
from bones.warhammer import AttackCounter, AttackPMF


class TestAttackCounterInit:
    def test_defaults(self) -> None:
        counter = AttackCounter(0)
        assert counter.attacks == 0
        assert counter.mortals == 0
        assert counter.wounds == ()

    def test_fields(self) -> None:
        # Test with all field parameters set.
        counter = AttackCounter(3, 2, (1,))
        assert counter.attacks == 3
        assert counter.mortals == 2
        assert counter.wounds == (1,)

    errors = [
        None,
        0.0,
        float("nan"),
        float("inf"),
        0 + 0j,
        "nope",
    ]


class TestAttackPMFInit:
    def test_defaults(self) -> None:
        pmf = AttackPMF()
        assert len(pmf) == 0

    def test_attack_copy(self) -> None:
        # Copy another AttackPMF.
        pmf1 = AttackPMF()
        pmf2 = AttackPMF(pmf1)
        assert pmf1.mapping == pmf2.mapping
        assert pmf1.total == pmf2.total

    def test_attack_from_iterable(self) -> None:
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(3)
        pmap: dict[AttackCounter, Weight] = {
            counter1: 3,
            counter2: 2,
            counter3: 1,
        }
        pmf = AttackPMF(Counter(pmap).elements())
        assert len(pmf) == 3
        assert pmf.weight(counter1) == pmap[counter1]
        assert pmf.weight(counter2) == pmap[counter2]
        assert pmf.weight(counter3) == pmap[counter3]

    def test_attack_from_dict(self) -> None:
        counter1 = AttackCounter(1)
        counter2 = AttackCounter(2)
        counter3 = AttackCounter(2)
        pmap: dict[AttackCounter, Weight] = {
            counter1: 3,
            counter2: 2,
            counter3: 1,
        }
        pmf = AttackPMF(pmap)
        assert len(pmf) == 2
        assert pmf.weight(counter1) == pmap[counter1]
        assert pmf.weight(counter2) == pmap[counter2]
        assert pmf.weight(counter3) == pmap[counter3]
