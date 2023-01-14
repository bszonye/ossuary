"""Unit tests for the bones.pmf module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import itertools
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, TypeAlias, TypeVar

import pytest

from bones.pmf import PMF, Weight

ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.
_T = TypeVar("_T")

WeightPair: TypeAlias = tuple[_T, Weight]
WeightPairs: TypeAlias = Sequence[WeightPair[_T]]


def expected(pairs: Iterable[tuple[_T, Weight]]) -> tuple[Counter[_T], Counter[_T]]:
    """Return expected PMF attributes for given weight pairs."""
    chain = itertools.chain.from_iterable([ev] * wt for ev, wt in pairs)
    counter = Counter(chain)
    gcd = math.gcd(*counter.values())
    norm = Counter({ev: wt // gcd for ev, wt in counter.items()})
    return counter, norm


class SubPMF(PMF[ET_co]):
    """Subclass for testing type hierarchy stuff."""


class TestPMFInit:
    def test_default(self) -> None:
        pairs: WeightPairs[Any] = []
        counter, norm = expected(pairs)

        pmf = PMF[Any]()

        assert len(pmf) == len(counter) == len(norm)
        assert pmf.mapping == dict(counter) == dict(norm)
        assert pmf.total == counter.total() == norm.total()

    def test_copy_default(self) -> None:
        # Copy another PMF (with normalization).
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1)

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(norm)
        assert pmf2.mapping == dict(norm)
        assert pmf2.total == norm.total()

        assert pmf1.mapping is not pmf2.mapping

    def test_copy_normalized(self) -> None:
        # Copy another PMF (with normalization).
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1, normalize=True)

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(norm)
        assert pmf2.mapping == dict(norm)
        assert pmf2.total == norm.total()

        assert pmf1.mapping is not pmf2.mapping

    def test_copy_exact(self) -> None:
        # Copy another PMF (without normalization).
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1, normalize=False)

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(counter)
        assert pmf2.mapping == dict(counter)
        assert pmf2.total == counter.total()

        assert pmf1.mapping is pmf2.mapping

    def test_normalize_default(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter))

        assert len(pmf) == len(norm)
        assert pmf.mapping == dict(norm)
        assert pmf.total == norm.total()

    def test_normalize_true(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=True)

        assert len(pmf) == len(norm)
        assert pmf.mapping == dict(norm)
        assert pmf.total == norm.total()

    def test_normalize_false(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)

        assert len(pmf) == len(counter)
        assert pmf.mapping == dict(counter)
        assert pmf.total == counter.total()

    def test_dict(self) -> None:
        pairs = [(0, 1), (3, 2), (7, 1)]
        counter, norm = expected(pairs)

        pmf = PMF(dict(counter))

        assert len(pmf) == len(counter) == len(norm)
        assert pmf.mapping == dict(counter) == dict(norm)
        assert pmf.total == counter.total() == norm.total()

    def test_counter(self) -> None:
        pairs = [(0, 1), (3, 2), (7, 1)]
        counter, norm = expected(pairs)

        pmf = PMF(counter)

        assert len(pmf) == len(counter) == len(norm)
        assert pmf.mapping == dict(counter) == dict(norm)
        assert pmf.total == counter.total() == norm.total()

    def test_iterable(self) -> None:
        elements = (0, 7, 3, 5, 7, 5, 3)
        counter = Counter(elements)

        pmf = PMF(elements)

        assert len(pmf) == len(counter)
        assert pmf.mapping == dict(counter)
        assert pmf.total == counter.total()

        # Domain order should match input order.
        assert pmf.domain == elements[: len(pmf.domain)]

    def test_pzero(self) -> None:
        # Test a PMF with a zero probability.
        mapping = {1: 2, 2: 0, 3: 1}
        counter = Counter(mapping)

        pmf = PMF(mapping)

        assert len(pmf) == len(mapping)
        assert pmf.mapping == mapping
        assert pmf.total == counter.total()

        assert pmf.domain == (1, 2, 3)
        assert pmf.support == (1, 3)

    type_errors: Any = (
        0,  # not iterable
        {0: "foo"},  # not a probability
        {0: Fraction(1)},  # not a probability
        ([],),  # not hashable
        ({},),  # not hashable
    )

    @pytest.mark.parametrize("error", type_errors)
    def test_type_error(self, error: Any) -> None:
        with pytest.raises(TypeError):
            PMF(error)

    def test_value_error(self) -> None:
        with pytest.raises(ValueError):
            PMF({0: -1})


class TestPMFFromPairs:
    def test_empty(self) -> None:
        items = ()
        pmf = PMF[Any]._from_pairs(items)  # pyright: ignore[reportPrivateUsage]
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 0

    def test_dict_items(self) -> None:
        items = {0: 1, 3: 2, 7: 1}
        pmf = PMF._from_pairs(items.items())  # pyright: ignore[reportPrivateUsage]
        assert len(pmf) == len(items)
        assert pmf.mapping == items
        assert pmf.total == sum(items.values())


class TestPMFFromIterable:
    def test_empty(self) -> None:
        items = ()
        pmf = PMF[Any]._from_iterable(items)  # pyright: ignore[reportPrivateUsage]
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 0

    def test_sequence(self) -> None:
        items = (1, 3, 1, 2, 1, 2)
        pmf = PMF._from_iterable(items)  # pyright: ignore[reportPrivateUsage]
        assert len(pmf) == 3
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}
        assert pmf.total == len(items)
        assert pmf.domain == (1, 3, 2)


class TestPMFConvert:
    def test_same(self) -> None:
        items = {0: 3}
        pmf1 = PMF(items, normalize=False)
        pmf2 = PMF[int].convert(pmf1)
        assert pmf1.mapping is pmf2.mapping
        assert pmf1.total is pmf2.total


class TestPMFNormalized:
    weights = {
        # Zero weights.
        (): 0,  # No weights.
        (0,): 0,  # No non-zero weights.
        (0, 0): 0,  # Multiple zero weights.
        # Integral weights with gcd == 1.
        (1,): 1,
        (1, 2): 3,
        (2, 3, 4): 9,
        # Integral weights with gcd > 1.
        (2, 4, 6): 6,
        (10, 15, 20): 9,
        (6, 12, 6, 12, 24): 10,
    }

    @pytest.mark.parametrize("weights, int_weight", weights.items())
    def test_normalized_default(
        self, weights: Sequence[Weight], int_weight: int
    ) -> None:
        # Test the default parameters with various item weights.
        items = {i: weights[i] for i in range(len(weights))}
        pmf = PMF(items)
        npmf = pmf.normalized()
        assert len(npmf) == len(pmf)
        assert npmf.total == int_weight

        # All weights should be integers after default normalization.
        weights = tuple(npmf.weights)
        for w in weights:
            assert isinstance(w, int)

        total = sum(weights)
        if total:
            assert npmf.total == total
            assert math.gcd(*weights) == 1
        else:
            assert len(npmf.support) == 0


class TestPMFUnaryOperator:
    ppos = (1, 2, 3)
    pneg = (-3, -2, -1)
    pmix = (-2, -1, 0, 1, 2)

    @pytest.mark.parametrize(
        "events,expect",
        (
            (ppos, ((-1, 1), (-2, 1), (-3, 1))),
            (pneg, ((3, 1), (2, 1), (1, 1))),
            (pmix, ((2, 1), (1, 1), (0, 1), (-1, 1), (-2, 1))),
        ),
    )
    def test_neg(self, events: Sequence[Any], expect: Sequence[Any]) -> None:
        pmf = -PMF(events)
        assert pmf.pairs == expect

    @pytest.mark.parametrize(
        "events,expect",
        (
            (ppos, ((1, 1), (2, 1), (3, 1))),
            (pneg, ((-3, 1), (-2, 1), (-1, 1))),
            (pmix, ((-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1))),
        ),
    )
    def test_pos(self, events: Sequence[Any], expect: Sequence[Any]) -> None:
        pmf = +PMF(events)
        assert pmf.pairs == expect

    @pytest.mark.parametrize(
        "events,expect",
        (
            (ppos, ((1, 1), (2, 1), (3, 1))),
            (pneg, ((3, 1), (2, 1), (1, 1))),
            (pmix, ((2, 2), (1, 2), (0, 1))),
        ),
    )
    def test_abs(self, events: Sequence[Any], expect: Sequence[Any]) -> None:
        pmf = abs(PMF(events))
        assert pmf.pairs == expect

    @pytest.mark.parametrize(
        "events,expect",
        (
            (ppos, ((-2, 1), (-3, 1), (-4, 1))),
            (pneg, ((2, 1), (1, 1), (0, 1))),
            (pmix, ((1, 1), (0, 1), (-1, 1), (-2, 1), (-3, 1))),
        ),
    )
    def test_invert(self, events: Sequence[Any], expect: Sequence[Any]) -> None:
        pmf = ~PMF(events)
        assert pmf.pairs == expect

    def test_round(self) -> None:
        pfloat = tuple(x / 8 for x in range(-12, 13))
        pmf = PMF(pfloat)
        assert round(pmf).pairs == (
            (-2, 1),
            (-1, 7),
            (0, 9),
            (1, 7),
            (2, 1),
        )

    def test_round_n(self) -> None:
        pfives = tuple(x * 5 for x in range(11))
        pmf = PMF(pfives)
        assert round(pmf, -1).pairs == (
            (0, 2),
            (10, 1),
            (20, 3),
            (30, 1),
            (40, 3),
            (50, 1),
        )

    def test_trunc(self) -> None:
        pfloat = tuple(x / 8 for x in range(-12, 13))
        pmf = PMF(pfloat)
        assert math.trunc(pmf).pairs == (
            (-1, 1),
            (0, 3),
            (1, 1),
        )

    def test_floor(self) -> None:
        pfloat = tuple(x / 8 for x in range(-12, 13))
        pmf = PMF(pfloat)
        assert math.floor(pmf).pairs == (
            (-2, 4),
            (-1, 8),
            (0, 8),
            (1, 5),
        )

    def test_ceil(self) -> None:
        pfloat = tuple(x / 8 for x in range(-12, 13))
        pmf = PMF(pfloat)
        assert math.ceil(pmf).pairs == (
            (-1, 5),
            (0, 8),
            (1, 8),
            (2, 4),
        )


class TestPMFBinaryOperator:
    def test_matmul(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (2 @ d3).pairs == (
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 2),
            (6, 1),
        )
        assert (d3 @ 2).pairs == (
            (2, 1),
            (4, 1),
            (6, 1),
        )
        p1 = 1 @ d3
        p2 = 2 @ d3
        p3 = 3 @ d3
        d3d3 = d3 @ d3
        assert d3d3.support == (1, 2, 3, 4, 5, 6, 7, 8, 9)
        for v, p in d3d3.graph:
            assert p == p1(v) / 3 + p2(v) / 3 + p3(v) / 3

    def test_add(self) -> None:
        pass  # TODO

    def test_sub(self) -> None:
        pass  # TODO

    def test_mul(self) -> None:
        pass  # TODO

    def test_truediv(self) -> None:
        pass  # TODO

    def test_floordiv(self) -> None:
        pass  # TODO

    def test_mod(self) -> None:
        pass  # TODO

    def test_pow(self) -> None:
        pass  # TODO

    def test_lshift(self) -> None:
        pass  # TODO

    def test_rshift(self) -> None:
        pass  # TODO

    def test_and(self) -> None:
        pass  # TODO

    def test_xor(self) -> None:
        pass  # TODO

    def test_or(self) -> None:
        pass  # TODO
