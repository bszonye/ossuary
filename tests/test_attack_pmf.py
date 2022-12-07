"""Unit tests for bones.warhammer.AttackPMF class."""

import lea
import lea.leaf
import pytest

from bones.warhammer import AttackCounter, AttackPMF, AttackSingle


class TestAttackPMF:
    """Test the AttackPMF constructor."""

    def test_default(self) -> None:
        """Test with default arguments."""
        attack = AttackPMF()
        pmf = attack.pmf.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (AttackCounter(1), 1)

    def test_attack_int(self) -> None:
        """Test with an int parameter."""
        attack = AttackPMF(1)
        pmf = attack.pmf.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (AttackCounter(1), 1)

    def test_attack_counter(self) -> None:
        """Test with a counter parameter."""
        counter = AttackCounter(4, 3, 2, 1)
        attack = AttackPMF(counter)
        pmf = attack.pmf.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (counter, 1)

    def test_attack_lea(self) -> None:
        """Test with a Lea parameter."""
        counter = AttackCounter(4, 3, 2, 1)
        pmf_in = lea.vals(counter)
        attack = AttackPMF(pmf_in)
        pmf_out = attack.pmf
        assert pmf_out.pmf_tuple == pmf_in.pmf_tuple

    def test_attack_single(self) -> None:
        """Test the default value singleton."""
        pmf = AttackSingle.pmf.pmf_tuple
        assert len(pmf) == 1
        assert pmf[0] == (AttackCounter(1), 1)

    def test_type_error(self) -> None:
        """Test parameter type errors."""
        with pytest.raises(TypeError):
            AttackPMF(lea.leaf.D6)
        with pytest.raises(TypeError):
            AttackPMF("nope")  # type: ignore
