"""Unit tests for the Characteristic class & subtypes."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import pytest

from bones.roll import d6
from bones.warhammer import RandomizableValue


class TestRandomizableValueFactory:
    def test_int(self) -> None:
        RandomizableValue.factory(0)
        # TODO: design this API

    def test_float(self) -> None:
        RandomizableValue.factory(0.0)
        # TODO: design this API

    def test_str(self) -> None:
        RandomizableValue.factory("1d6")
        # TODO: design this API

    def test_pmf(self) -> None:
        RandomizableValue.factory(d6)
        # TODO: design this API

    def test_error(self) -> None:
        with pytest.raises(TypeError):
            RandomizableValue.factory({})  # type: ignore
