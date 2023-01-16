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

        assert pmf1.mapping is pmf2.mapping  # copy should share data

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(counter)
        assert pmf2.mapping == dict(counter)
        assert pmf2.total == counter.total()

    def test_copy_exact(self) -> None:
        # Copy another PMF (without normalization).
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1, normalize=False)

        assert pmf1.mapping is pmf2.mapping  # copy should share data

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf1) == len(counter) == len(pmf2)
        assert pmf1.mapping == dict(counter) == pmf2.mapping
        assert pmf1.total == counter.total() == pmf2.total

    def test_copy_normalized(self) -> None:
        # Copy another PMF (with normalization).
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1, normalize=True)

        assert pmf1.mapping is not pmf2.mapping  # copy should NOT share data

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(norm)
        assert pmf2.mapping == dict(norm)
        assert pmf2.total == norm.total()

    def test_copy_irreducible(self) -> None:
        # Copy a PMF that's irreducible (already in normal form).
        pairs = [(1, 1), (2, 2), (3, 3)]
        counter, norm = expected(pairs)
        assert counter == norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF(pmf1, normalize=True)

        assert pmf1.mapping is pmf2.mapping  # copy should share data

        assert len(pmf1) == len(counter) == len(pmf2)
        assert pmf1.mapping == dict(counter) == pmf2.mapping
        assert pmf1.total == counter.total() == pmf2.total

    def test_normalize_default(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter))

        assert len(pmf) == len(counter)
        assert pmf.mapping == dict(counter)
        assert pmf.total == counter.total()

    def test_normalize_false(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)

        assert len(pmf) == len(counter)
        assert pmf.mapping == dict(counter)
        assert pmf.total == counter.total()

    def test_normalize_true(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=True)

        assert len(pmf) == len(norm)
        assert pmf.mapping == dict(norm)
        assert pmf.total == norm.total()

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


class TestPMFFromSelf:
    def test_subtypes(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        # Chain constructors up & down subtypes.
        pmf1 = SubPMF(norm)
        pmf2 = SubPMF.from_self(pmf1)
        pmf3 = PMF.from_self(pmf2)
        pmf4 = PMF.from_self(pmf3)
        pmf5 = SubPMF.from_self(pmf4)

        # Check all of the types.
        assert type(pmf1) is SubPMF
        assert type(pmf2) is SubPMF
        assert type(pmf3) is PMF
        assert type(pmf4) is PMF
        assert type(pmf5) is SubPMF

        # Only the same-type copies should share data.
        assert pmf1.mapping is pmf2.mapping
        assert pmf2.mapping is not pmf3.mapping
        assert pmf3.mapping is pmf4.mapping
        assert pmf4.mapping is not pmf5.mapping

        # Check attributes.
        for pmf in (pmf1, pmf2, pmf3, pmf4, pmf5):
            assert len(pmf) == len(norm)
            assert pmf.mapping == dict(norm)
            assert pmf.total == norm.total()

    def test_normalize_false(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        # Chain constructors up & down subtypes.
        pmf1 = SubPMF(counter, normalize=False)
        pmf2 = SubPMF.from_self(pmf1, normalize=False)
        pmf3 = PMF.from_self(pmf2, normalize=False)
        pmf4 = PMF.from_self(pmf3, normalize=False)
        pmf5 = SubPMF.from_self(pmf4, normalize=False)

        # Only the same-type copies should share data.
        assert pmf1.mapping is pmf2.mapping
        assert pmf2.mapping is not pmf3.mapping
        assert pmf3.mapping is pmf4.mapping
        assert pmf4.mapping is not pmf5.mapping

        # Check attributes.
        for pmf in (pmf1, pmf2, pmf3, pmf4, pmf5):
            assert len(pmf) == len(counter)
            assert pmf.mapping == dict(counter)
            assert pmf.total == counter.total()

    def test_normalize_true(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        # Mix and match subtypes.
        pmf1a = PMF(counter, normalize=False)
        pmf2a = PMF.from_self(pmf1a, normalize=True)
        pmf3a = SubPMF.from_self(pmf1a, normalize=True)
        pmf1b = PMF(counter, normalize=False)
        pmf2b = PMF.from_self(pmf1b, normalize=True)
        pmf3b = SubPMF.from_self(pmf1b, normalize=True)

        # None of the copies should share data.
        assert pmf1a.mapping is not pmf2a.mapping
        assert pmf1a.mapping is not pmf3a.mapping
        assert pmf1b.mapping is not pmf2b.mapping
        assert pmf1b.mapping is not pmf3b.mapping

        # Originals should have the original data.
        for pmf in (pmf1a, pmf1b):
            assert len(pmf) == len(counter)
            assert pmf.mapping == dict(counter)
            assert pmf.total == counter.total()
        # Copies should have the normalized data.
        for pmf in (pmf2a, pmf3a, pmf2b, pmf3b):
            assert len(pmf) == len(norm)
            assert pmf.mapping == dict(norm)
            assert pmf.total == norm.total()


class TestPMFFromPairs:
    def test_empty(self) -> None:
        items = ()
        pmf = PMF[Any].from_pairs(items)
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 0

    def test_dict_items(self) -> None:
        items = {0: 1, 3: 2, 7: 1}
        pmf = PMF.from_pairs(items.items())
        assert len(pmf) == len(items)
        assert pmf.mapping == items
        assert pmf.total == sum(items.values())


class TestPMFFromIterable:
    def test_empty(self) -> None:
        items = ()
        pmf = PMF[Any].from_iterable(items)
        assert len(pmf) == 0
        assert pmf.mapping == {}
        assert pmf.total == 0

    def test_sequence(self) -> None:
        items = (1, 3, 1, 2, 1, 2)
        pmf = PMF.from_iterable(items)
        assert len(pmf) == 3
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}
        assert pmf.total == len(items)
        assert pmf.domain == (1, 3, 2)


class TestPMFConvert:
    def test_same(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = PMF[int].convert(pmf1)

        assert pmf1 is pmf2  # convert should be a no-op

        assert len(pmf1) == len(counter) == len(pmf2)
        assert pmf1.mapping == dict(counter) == pmf2.mapping
        assert pmf1.total == counter.total() == pmf2.total

    def test_subtype(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = PMF(counter, normalize=False)
        pmf2 = SubPMF[int].convert(pmf1)

        assert type(pmf2) is SubPMF
        assert pmf1.mapping is not pmf2.mapping  # copy should NOT share data

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(counter)
        assert pmf2.mapping == dict(counter)
        assert pmf2.total == counter.total()

    def test_supertype(self) -> None:
        pairs = [(1, 3), (2, 6)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf1 = SubPMF(counter, normalize=False)
        pmf2 = PMF[int].convert(pmf1)

        assert type(pmf2) is PMF
        assert pmf1.mapping is not pmf2.mapping  # copy should NOT share data

        assert len(pmf1) == len(counter)
        assert pmf1.mapping == dict(counter)
        assert pmf1.total == counter.total()

        assert len(pmf2) == len(counter)
        assert pmf2.mapping == dict(counter)
        assert pmf2.total == counter.total()


class TestPMFAccessors:
    def test_empty(self) -> None:
        pmf = PMF[Any]()

        # Test all of the instance properties.
        assert pmf.mapping == {}
        assert pmf.total == 0
        assert pmf.gcd == 0
        assert pmf.domain == ()
        assert pmf.support == ()
        assert pmf.weights == ()
        assert pmf.sum_weights == ()
        assert pmf.tail_weights == ()
        assert pmf.pairs == ()
        assert pmf.image == ()
        assert pmf.graph == ()

    def test_properties(self) -> None:
        mapping = {1: 8, 2: 6, 3: 4, 4: 2, 5: 0}
        total = sum(mapping.values())
        gcd = math.gcd(*mapping.values())
        domain = tuple(mapping.keys())
        support = tuple(ev for ev, wt in mapping.items() if wt)
        weights = tuple(mapping.values())
        sum_weights = tuple(itertools.accumulate(mapping.values()))
        tail_weights = tuple(
            total - wt for wt in itertools.accumulate(mapping.values())
        )
        pairs = tuple((ev, wt) for ev, wt in mapping.items())
        image = tuple(Fraction(wt, total) for wt in mapping.values())
        graph = tuple((ev, Fraction(wt, total)) for ev, wt in mapping.items())

        pmf = PMF(mapping, normalize=False)

        # Test all of the instance properties.
        assert pmf.mapping == mapping
        assert pmf.total == total
        assert pmf.gcd == gcd
        assert pmf.domain == domain
        assert pmf.support == support
        assert pmf.weights == weights
        assert pmf.sum_weights == sum_weights
        assert pmf.tail_weights == tail_weights
        assert pmf.pairs == pairs
        assert pmf.image == image
        assert pmf.graph == graph

    def test_calls(self) -> None:
        mapping = {1: 8, 2: 6, 3: 4, 4: 2, 5: 0}

        pmf = PMF(mapping, normalize=False)

        # Test the callable interface and its underlying methods.
        for ev, wt in mapping.items():
            assert pmf.weight(ev) == wt
            assert pmf.probability(ev) == Fraction(wt, pmf.total)
            assert pmf(ev) == Fraction(wt, pmf.total)

    def test_zeroes(self) -> None:
        mapping = {1: 8, 2: 6, 3: 4, 4: 2, 5: 0}

        pmf = PMF(mapping, normalize=False)

        # Test values that should have zero probability.
        zeroes: Sequence[Any] = (
            5,  # explicitly zero in the mapping
            6,  # not in the mapping
            "nope",  # not an integer
            list(),  # not even hashable
        )
        for ev in zeroes:
            assert pmf.weight(ev) == 0
            assert pmf.probability(ev) == 0
            assert pmf(ev) == 0


class TestPMFCopy:
    # Test the copying methods: copy, normalized, sorted.

    def test_copy_normalize_false(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        copy = pmf.copy(normalize=False)  # exact

        # copy-specific checks
        assert not copy.is_normal()
        assert copy.mapping is pmf.mapping
        # general checks
        assert len(copy) == len(counter)
        assert copy.mapping == dict(counter)
        assert copy.total == counter.total()

    def test_copy_normalize_true(self) -> None:
        pairs = [(0, 2), (3, 4), (7, 2)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        copy = pmf.copy(normalize=True)  # normalized

        # copy-specific checks
        assert copy.is_normal()
        assert copy.mapping is not pmf.mapping
        # general checks
        assert len(copy) == len(norm)
        assert copy.mapping == dict(norm)
        assert copy.total == norm.total()

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
    def test_normalized(self, weights: Sequence[Weight], int_weight: int) -> None:
        # Test the default parameters with various item weights.
        items = {i: weights[i] for i in range(len(weights))}

        pmf = PMF(items, normalize=False)
        normalized = pmf.normalized()

        assert normalized.is_normal()
        assert len(normalized) == len(pmf)
        assert normalized.total == int_weight

        # All weights should be integers after default normalization.
        weights = tuple(normalized.weights)
        for w in weights:
            assert isinstance(w, int)

        total = sum(weights)
        if total:
            assert normalized.total == total
            assert math.gcd(*weights) == 1
        else:
            assert len(normalized.support) == 0

    def test_sorted_normalize_false(self) -> None:
        pairs = [("a", 2), ("lazy", 4), ("black", 6), ("cat", 8)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        assert not pmf.is_sorted()

        sort = pmf.sorted(normalize=False)  # exact
        assert sort.domain == ("a", "black", "cat", "lazy")
        assert sort.is_sorted()
        assert not sort.is_normal()
        # general checks
        assert len(sort) == len(counter)
        assert sort.mapping == dict(counter)
        assert sort.total == counter.total()

    def test_sorted_normalize_true(self) -> None:
        pairs = [("a", 2), ("lazy", 4), ("black", 6), ("cat", 8)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        assert not pmf.is_sorted()

        sort = pmf.sorted(normalize=True)  # normalized
        assert sort.domain == ("a", "black", "cat", "lazy")
        assert sort.is_sorted()
        assert sort.is_normal()
        # general checks
        assert len(sort) == len(norm)
        assert sort.mapping == dict(norm)
        assert sort.total == norm.total()

    def test_sorted_key(self) -> None:
        pairs = [("a", 1), ("lazy", 2), ("black", 3), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter == norm

        pmf = PMF(dict(counter))
        assert not pmf.is_sorted(key=len)

        sort = pmf.sorted(key=len)
        assert sort.domain == ("a", "cat", "lazy", "black")
        assert sort.is_sorted(key=len)
        # general checks
        assert len(sort) == len(norm)
        assert sort.mapping == dict(norm)
        assert sort.total == norm.total()

    def test_sorted_reverse(self) -> None:
        pairs = [("a", 1), ("lazy", 2), ("black", 3), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter == norm

        pmf = PMF(dict(counter))
        assert not pmf.is_sorted(key=len)

        sort = pmf.sorted(reverse=True)
        assert sort.domain == ("lazy", "cat", "black", "a")
        assert sort.is_sorted(reverse=True)
        # general checks
        assert len(sort) == len(norm)
        assert sort.mapping == dict(norm)
        assert sort.total == norm.total()

    def test_sorted_key_reverse(self) -> None:
        pairs = [("a", 1), ("lazy", 2), ("black", 3), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter == norm

        pmf = PMF(dict(counter))
        assert not pmf.is_sorted(key=len, reverse=True)

        sort = pmf.sorted(key=len, reverse=True)
        assert sort.domain == ("black", "lazy", "cat", "a")
        assert sort.is_sorted(key=len, reverse=True)
        # general checks
        assert len(sort) == len(norm)
        assert sort.mapping == dict(norm)
        assert sort.total == norm.total()

    def test_short_sorts(self) -> None:
        assert PMF().is_sorted()
        assert PMF((1,)).is_sorted()


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
            (-1, 5),
            (0, 15),
            (1, 5),
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
