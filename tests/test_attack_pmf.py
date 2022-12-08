"""Unit tests for bones.warhammer.AttackPMF class."""

import lea
import lea.leaf
import pytest

from bones.warhammer import AttackCounter, AttackPMF


class TestAttackPMF:
    """Test the AttackPMF constructor."""

    def test_default(self) -> None:
        """Test with default arguments."""
        attack = AttackPMF()
        pmf = attack.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (AttackCounter(1), 1)
        print(hash(attack))
        print(hash(attack.pmf))

    def test_attack_int(self) -> None:
        """Test with an int parameter."""
        attack = AttackPMF(1)
        pmf = attack.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (AttackCounter(1), 1)

    def test_attack_counter(self) -> None:
        """Test with a counter parameter."""
        counter = AttackCounter(4, 3, 2, 1)
        attack = AttackPMF(counter)
        pmf = attack.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (counter, 1)

    def test_attack_lea(self) -> None:
        """Test with a Lea parameter."""
        pmf = lea.vals(AttackCounter(4, 3, 2, 1))
        attack = AttackPMF(pmf)
        assert attack.pmf_tuple == pmf.pmf_tuple

    def test_type_error(self) -> None:
        """Test parameter type errors."""
        with pytest.raises(TypeError):
            AttackPMF(lea.leaf.D6)
        with pytest.raises(TypeError):
            AttackPMF("nope")  # type: ignore
