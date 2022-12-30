"""Unit tests for bones.warhammer.Characteristic class & subtypes."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

from bones.pmf import D6
from bones.warhammer import RandomizableValue


class TestRandomizableValueFactory:
    """Test the RandomizableValue.factory method."""

    def test_int(self) -> None:
        """Test with int argument."""
        RandomizableValue.factory(0)
        # TODO: design this API

    def test_float(self) -> None:
        """Test with float argument."""
        RandomizableValue.factory(0.0)
        # TODO: design this API

    def test_str(self) -> None:
        """Test with str argument."""
        RandomizableValue.factory("1d6")
        # TODO: design this API

    def test_pmf(self) -> None:
        """Test with pmf argument."""
        RandomizableValue.factory(D6)
        # TODO: design this API

    def test_error(self) -> None:
        """Test with invalie type."""
        with pytest.raises(TypeError):
            RandomizableValue.factory({})  # type: ignore
