"""Unit tests for the bones.pmf module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import importlib.util
import itertools
import math
import operator
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from typing import Any, TypeAlias, TypeVar
from unittest.mock import patch

import pytest
from pytest import approx  # pyright: ignore[reportUnknownVariableType]

from bones.pmf import multiset_comb, multiset_perm, PMF, QK, quantile_name, Weight

ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.
_T = TypeVar("_T")

WeightPair: TypeAlias = tuple[_T, Weight]
WeightPairs: TypeAlias = Sequence[WeightPair[_T]]
Cap: TypeAlias = pytest.CaptureFixture[str]


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

    def test_reverse(self) -> None:
        pmf1 = PMF((1, 2, 3))
        pmf2 = PMF.from_self(pmf1, reverse=True)
        assert len(pmf1) == len(pmf2)
        assert pmf1.mapping == pmf2.mapping
        assert pmf1.total == pmf2.total
        assert pmf1.domain == tuple(reversed(pmf2.domain))


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

    def test_reverse(self) -> None:
        items = {0: 1, 3: 2, 7: 1}
        pmf = PMF.from_pairs(items.items(), reverse=True)
        assert len(pmf) == len(items)
        assert pmf.mapping == items
        assert pmf.total == sum(items.values())
        assert pmf.domain == (7, 3, 0)


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

    def test_reverse(self) -> None:
        items = (1, 3, 1, 2, 1, 2)
        pmf = PMF.from_iterable(items, reverse=True)
        assert len(pmf) == 3
        assert pmf.mapping == {1: 3, 2: 2, 3: 1}
        assert pmf.total == len(items)
        assert pmf.domain == (2, 1, 3)


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
        assert pmf.zeroes == ()
        assert pmf.weights == ()
        assert pmf.sum_weights == ()
        assert pmf.tail_weights == ()
        assert pmf.pairs == ()
        assert pmf.ranked_pairs == ()
        assert pmf.image == ()
        assert pmf.graph == ()

    def test_properties(self) -> None:
        mapping = {1: 8, 2: 6, 3: 4, 4: 2, 5: 0}
        total = sum(mapping.values())
        gcd = math.gcd(*mapping.values())
        domain = tuple(mapping.keys())
        support = tuple(ev for ev, wt in mapping.items() if wt)
        zeroes = tuple(ev for ev, wt in mapping.items() if not wt)
        weights = tuple(mapping.values())
        sum_weights = tuple(itertools.accumulate(mapping.values()))
        tail_weights = tuple(
            total - wt for wt in itertools.accumulate(mapping.values())
        )
        pairs = tuple((ev, wt) for ev, wt in mapping.items())
        ranked_pairs = tuple(sorted(mapping.items(), key=(lambda pair: pair[1])))
        image = tuple(Fraction(wt, total) for wt in mapping.values())
        graph = tuple((ev, Fraction(wt, total)) for ev, wt in mapping.items())

        pmf = PMF(mapping, normalize=False)

        # Test all of the instance properties.
        assert pmf.mapping == mapping
        assert pmf.total == total
        assert pmf.gcd == gcd
        assert pmf.domain == domain
        assert pmf.support == support
        assert pmf.zeroes == zeroes
        assert pmf.weights == weights
        assert pmf.sum_weights == sum_weights
        assert pmf.tail_weights == tail_weights
        assert pmf.pairs == pairs
        assert pmf.ranked_pairs == ranked_pairs
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

    def test_copy_cache(self) -> None:
        pets = ("cat", "dog", "cat", "ferret", "cat", "goat")

        # Initialize cached properties.
        pmf = PMF(pets)
        _ = pmf.domain
        _ = pmf.support
        _ = pmf.zeroes
        _ = pmf.weights
        _ = pmf.sum_weights
        _ = pmf.tail_weights
        _ = pmf.pairs
        _ = pmf.ranked_pairs
        _ = pmf.image
        _ = pmf.graph

        # Verify exact copies.
        copy = pmf.copy()
        assert copy.mapping is pmf.mapping
        assert copy.total is pmf.total
        assert copy.gcd is pmf.gcd
        assert copy.domain is pmf.domain
        assert copy.support is pmf.support
        assert copy.zeroes is pmf.zeroes
        assert copy.weights is pmf.weights
        assert copy.sum_weights is pmf.sum_weights
        assert copy.tail_weights is pmf.tail_weights
        assert copy.pairs is pmf.pairs
        assert copy.ranked_pairs is pmf.ranked_pairs
        assert copy.image is pmf.image
        assert copy.graph is pmf.graph

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

    def test_ranked_normalize_false(self) -> None:
        pairs = [("a", 8), ("lazy", 6), ("black", 2), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        assert not pmf.is_ranked()

        sort = pmf.ranked(normalize=False)  # exact
        assert sort.domain == ("black", "cat", "lazy", "a")
        assert sort.is_ranked()
        assert not sort.is_normal()
        # general checks
        assert len(sort) == len(counter)
        assert sort.mapping == dict(counter)
        assert sort.total == counter.total()

    def test_ranked_normalize_true(self) -> None:
        pairs = [("a", 8), ("lazy", 6), ("black", 2), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        assert not pmf.is_ranked()

        sort = pmf.ranked(normalize=True)  # exact
        assert sort.domain == ("black", "cat", "lazy", "a")
        assert sort.is_ranked()
        assert sort.is_normal()
        # general checks
        assert len(sort) == len(norm)
        assert sort.mapping == dict(norm)
        assert sort.total == norm.total()

    def test_ranked_reverse(self) -> None:
        pairs = [("a", 8), ("lazy", 6), ("black", 2), ("cat", 4)]
        counter, norm = expected(pairs)
        assert counter != norm

        pmf = PMF(dict(counter), normalize=False)
        assert not pmf.is_ranked(reverse=True)

        sort = pmf.ranked(reverse=True)  # exact
        assert sort.domain == ("a", "lazy", "cat", "black")
        assert sort.is_ranked(reverse=True)
        assert not sort.is_normal()
        # general checks
        assert len(sort) == len(counter)
        assert sort.mapping == dict(counter)
        assert sort.total == counter.total()

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
        assert not pmf.is_sorted(reverse=True)

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
        assert PMF().is_ranked()
        assert PMF((1,)).is_ranked()
        assert PMF().is_sorted()
        assert PMF((1,)).is_sorted()


class TestPMFStatistics:
    def test_mean(self) -> None:
        d6 = PMF(range(1, 7))
        assert d6.exact_mean == Fraction(7, 2)
        assert d6.mean == 3.5
        with pytest.raises(ZeroDivisionError):
            PMF().exact_mean
        with pytest.raises(ZeroDivisionError):
            PMF().mean

    def test_variance(self) -> None:
        d6 = PMF(range(1, 7))
        assert d6.exact_variance == Fraction(35, 12)
        assert d6.variance == approx(35 / 12)
        with pytest.raises(ZeroDivisionError):
            PMF().exact_variance
        with pytest.raises(ZeroDivisionError):
            PMF().variance

    def test_standard_deviation(self) -> None:
        d6 = PMF(range(1, 7))
        assert d6.standard_deviation == approx(math.sqrt(35 / 12))
        with pytest.raises(ZeroDivisionError):
            PMF().standard_deviation

    def test_multimode(self) -> None:
        d0 = PMF[int]()
        d6 = PMF(range(1, 7))
        assert d0.multimode == ()
        assert d6.multimode == (1, 2, 3, 4, 5, 6)
        assert (2 @ d6).multimode == (7,)
        assert (3 @ d6).multimode == (10, 11)

    def test_population(self) -> None:
        pmf = PMF({1: 3, 2: 2, 3: 1, 4: 0})
        pop = pmf.population()
        assert isinstance(pop, Iterator)
        assert tuple(pop) == (1, 1, 1, 2, 2, 3)

    def test_auto_quantile(self) -> None:
        # Check breakpoints in uniform distributions.
        assert PMF(range(0)).auto_quantile == 1
        assert PMF(range(6)).auto_quantile == 1
        assert PMF(range(7)).auto_quantile == 2
        assert PMF(range(14)).auto_quantile == 2
        assert PMF(range(15)).auto_quantile == 4
        assert PMF(range(18)).auto_quantile == 4
        assert PMF(range(19)).auto_quantile == 5
        assert PMF(range(100)).auto_quantile == 5
        # Check uneven distributions.
        assert (2 @ PMF(range(2))).auto_quantile == 1
        assert (2 @ PMF(range(3))).auto_quantile == 1
        assert (2 @ PMF(range(4))).auto_quantile == 2
        assert (2 @ PMF(range(6))).auto_quantile == 2
        assert (2 @ PMF(range(8))).auto_quantile == 4
        assert (2 @ PMF(range(10))).auto_quantile == 5
        assert (2 @ PMF(range(20))).auto_quantile == 5


class TestPMFOutput:
    def test_format_pairs_empty(self) -> None:
        pmf = PMF[Any]()
        pairs = pmf.format_pairs()
        assert isinstance(pairs, Iterator)
        assert tuple(pairs) == ()

    def test_format_pairs_hex(self) -> None:
        pmf = PMF((1, 1, 1, 2, 2, 3))
        pairs = pmf.format_pairs("#x")
        assert isinstance(pairs, Iterator)
        assert tuple(pairs) == (
            ("0x1", "0x3"),
            ("0x2", "0x2"),
            ("0x3", "0x1"),
        )

    def test_format_pairs_scale(self) -> None:
        pmf = PMF((1, 1, 1, 2, 2, 3))
        pairs = pmf.format_pairs(".2f", scale=100)
        assert isinstance(pairs, Iterator)
        assert tuple(pairs) == (
            ("1.00", "50.00"),
            ("2.00", "33.33"),
            ("3.00", "16.67"),
        )

    def test_plot_console(self, capsys: Cap) -> None:
        pmf = PMF((1, 2, 1, 3))
        pmf.plot(console=True)
        cap = capsys.readouterr()
        assert cap.out == "1  50.00\n2  25.00\n3  25.00\n"
        assert cap.err == ""

    def test_plot_fallback(self, capsys: Cap) -> None:
        pmf = PMF((1, 2, 1, 3))

        # Simulate a runtime environment without matplotlib.
        with patch("importlib.util.find_spec", return_value=None) as mock_find_spec:
            pmf.plot(console=False)
        mock_find_spec.assert_called_once_with("matplotlib")

        # Test the fallback console output.
        cap = capsys.readouterr()
        assert cap.out == "1  50.00\n2  25.00\n3  25.00\n"
        assert cap.err == ""

    def test_plot_ungrouped(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            return  # OK, ignore this test in a non-plot environment.

        pmf = PMF({1: 1, 2: 2, 3: 0, 4: 2, 5: 1})
        with (
            patch("matplotlib.axes.Axes.bar") as mock_bar,
            patch("matplotlib.axes.Axes.bar_label") as mock_bar_label,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            pmf.plot(q=1)

        assert mock_bar.call_count == 2
        mock_bar_label.assert_called_once()
        mock_show.assert_called_once_with(block=True)

    def test_plot_quantiles(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            return  # OK, ignore this test in a non-plot environment.

        pmf = PMF({1: 1, 2: 2, 3: 0, 4: 2, 5: 1})
        with (
            patch("matplotlib.axes.Axes.bar") as mock_bar,
            patch("matplotlib.axes.Axes.bar_label") as mock_bar_label,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            pmf.plot(q=2)

        assert mock_bar.call_count == 2
        mock_bar_label.assert_called_once()
        mock_show.assert_called_once_with(block=True)

    def test_plot_no_stats(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            return  # OK, ignore this test in a non-plot environment.

        pmf = PMF(("cat", "dog", "goat"))
        with (
            patch("matplotlib.axes.Axes.set_xlabel") as mock_set_xlabel,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            pmf.plot(q=1, stats=True)
            pmf.plot(q=1, stats=False)

        # This PMF shouldn't show stats either way.
        mock_set_xlabel.assert_not_called()
        assert mock_show.call_count == 2

    def test_tabulate_empty(self) -> None:
        pmf = PMF[Any]()
        tab = pmf.tabulate()
        assert isinstance(tab, Iterator)
        assert tuple(tab) == ()

    def test_tabulate_default(self) -> None:
        pmf = PMF((1, 2, 3))
        tab = pmf.tabulate()
        assert isinstance(tab, Iterator)
        assert tuple(tab) == (
            "1  1/3",
            "2  1/3",
            "3  1/3",
        )

    def test_tabulate_separator(self) -> None:
        pmf = PMF((1, 2))
        tab = pmf.tabulate(separator=" = ")
        assert isinstance(tab, Iterator)
        assert tuple(tab) == (
            "1 = 1/2",
            "2 = 1/2",
        )

    def test_tabulate_align_true(self) -> None:
        pmf = PMF((10, 9))
        tab = pmf.tabulate(align=True)
        assert isinstance(tab, Iterator)
        assert tuple(tab) == (
            "10  1/2",
            " 9  1/2",
        )

    def test_tabulate_align_false(self) -> None:
        pmf = PMF((10, 9))
        tab = pmf.tabulate(align=False)
        assert isinstance(tab, Iterator)
        assert tuple(tab) == (
            "10: 1/2",
            "9: 1/2",
        )

    def test_tabulate_text(self) -> None:
        pmf = PMF(("cat", "dog", "snake", "goat", "llama"))
        tab = pmf.tabulate()
        assert isinstance(tab, Iterator)
        assert tuple(tab) == (
            "cat    1/5",
            "dog    1/5",
            "snake  1/5",
            "goat   1/5",
            "llama  1/5",
        )

    def test_format_default(self) -> None:
        pets = ("cat", "dog")
        pmf = PMF(pets)
        assert format(pmf) == "{cat: 1/2, dog: 1/2}"

    def test_format_percent(self) -> None:
        pets = ("cat", "dog")
        pmf = PMF(pets)
        assert format(pmf, ":.0%") == "{cat: 50%, dog: 50%}"

    def test_format_hex(self) -> None:
        pmf = PMF((9, 10))
        assert format(pmf, "#0x:.2f") == "{0x9: 0.50, 0xa: 0.50}"

    def test_repr_empty(self) -> None:
        pmf = PMF[Any]()
        assert repr(pmf) == "PMF()"

    def test_repr_int(self) -> None:
        pmf = PMF((1, 1, 2))
        assert repr(pmf) == "PMF({1: 2, 2: 1})"

    def test_repr_str(self) -> None:
        pets = ("cat", "dog", "cat")
        pmf = PMF(pets)
        assert repr(pmf) == "PMF({'cat': 2, 'dog': 1})"

    def test_repr_gcd(self) -> None:
        pmf = PMF((1, 1, 1), normalize=False)
        assert repr(pmf) == "PMF({1: 3}, normalize=False)"

    def test_repr_subtype(self) -> None:
        pmf = SubPMF[Any]()
        assert repr(pmf) == "SubPMF()"

    def test_str(self) -> None:
        pets = ("cat", "dog")
        pmf = PMF(pets)
        assert str(pmf) == "{cat: 1/2, dog: 1/2}"


class TestPMFCombinatorics:
    def test_combinations_empty(self) -> None:
        pmf = PMF[Any]()
        assert pmf.combinations(1) == ()

    def test_combinations(self) -> None:
        d3 = PMF((1, 2, 3))
        assert d3.combinations(0) == ((),)
        assert d3.combinations(1) == ((1,), (2,), (3,))
        assert d3.combinations(2) == ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))

    def test_combinations_error(self) -> None:
        d3 = PMF((1, 2, 3))
        with pytest.raises(ValueError):
            d3.combinations(-1)

    def test_combination_weights(self) -> None:
        d3 = PMF((1, 2, 3))
        assert d3.combination_weights(0) == {(): 0}
        assert d3.combination_weights(1) == {(1,): 1, (2,): 1, (3,): 1}
        assert d3.combination_weights(2) == {
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 2,
            (2, 2): 1,
            (2, 3): 2,
            (3, 3): 1,
        }


class TestPMFHigherOrder:
    def test_map_string(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        pmf = PMF(pets)
        caps = pmf.map(str.upper)
        assert caps.domain == ("CAT", "DOG", "BIRD", "FISH", "SNAKE")
        assert caps.weights == pmf.weights

    def test_times(self) -> None:
        pmf = PMF((1, 2, 3))
        assert pmf.times(1) == pmf
        assert pmf.times(2).mapping == (pmf + pmf).mapping
        assert pmf.times(3, operator.sub).mapping == (pmf - pmf - pmf).mapping
        assert pmf.times(3, operator.sub).mapping != (pmf - (pmf - pmf)).mapping

    def test_rtimes(self) -> None:
        pmf = PMF((1, 2, 3))
        assert pmf.rtimes(1) == pmf
        assert pmf.rtimes(2).mapping == (pmf + pmf).mapping
        assert pmf.rtimes(3, operator.sub).mapping == (pmf - (pmf - pmf)).mapping
        assert pmf.rtimes(3, operator.sub).mapping != (pmf - pmf - pmf).mapping

    def test_times_error(self) -> None:
        pmf = PMF((1, 2, 3))
        with pytest.raises(ValueError):
            pmf.times(0)
        with pytest.raises(ValueError):
            pmf.rtimes(0)


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

    def test_rmatmul_errors(self) -> None:
        d3 = PMF((1, 2, 3))
        with pytest.raises(TypeError):
            _ = "cat" @ d3
        with pytest.raises(TypeError):
            _ = (1 + 0j) @ d3

    def test_add(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 + d3).mapping == {2: 1, 3: 2, 4: 3, 5: 2, 6: 1}
        assert (d3 + 1).mapping == {2: 1, 3: 1, 4: 1}
        assert (1 + d3).mapping == (d3 + 1).mapping

    def test_sub(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 - d3).mapping == {0: 3, -1: 2, -2: 1, 1: 2, 2: 1}
        assert (d3 - 1).mapping == {0: 1, 1: 1, 2: 1}
        assert (1 - d3).mapping == {0: 1, -1: 1, -2: 1}

    def test_mul(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 * d3).mapping == {1: 1, 2: 2, 3: 2, 4: 1, 6: 2, 9: 1}
        assert (d3 * 2).mapping == {2: 1, 4: 1, 6: 1}
        assert (2 * d3).mapping == (d3 * 2).mapping

    def test_truediv(self) -> None:
        pmf = PMF((0.5, 1.0, 2.0))
        assert (pmf / 2.0).mapping == {0.25: 1, 0.5: 1, 1.0: 1}
        assert (2.0 / pmf).mapping == {4.0: 1, 2.0: 1, 1.0: 1}

    def test_floordiv(self) -> None:
        pmf = PMF((6, 12, 24))
        assert (pmf // 3).mapping == {2: 1, 4: 1, 8: 1}
        assert (48 // pmf).mapping == {8: 1, 4: 1, 2: 1}

    def test_mod(self) -> None:
        pmf = PMF((2, 3, 4))
        assert (pmf % 3).mapping == {2: 1, 0: 1, 1: 1}
        assert (3 % pmf).mapping == {1: 1, 0: 1, 3: 1}

    def test_pow(self) -> None:
        pmf = PMF((2, 3, 4))
        assert (pmf**2).mapping == {4: 1, 9: 1, 16: 1}
        assert (2**pmf).mapping == {4: 1, 8: 1, 16: 1}

    def test_lshift(self) -> None:
        pmf = PMF((2, 3, 4))
        assert (pmf << 1).mapping == {4: 1, 6: 1, 8: 1}
        assert (3 << pmf).mapping == {12: 1, 24: 1, 48: 1}

    def test_rshift(self) -> None:
        pmf = PMF((2, 3, 4))
        assert (pmf >> 1).mapping == {1: 2, 2: 1}
        assert (27 >> pmf).mapping == {6: 1, 3: 1, 1: 1}

    def test_and(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 & d3).mapping == {1: 3, 0: 2, 2: 3, 3: 1}
        assert (d3 & 1).mapping == {1: 2, 0: 1}
        assert (1 & d3).mapping == (d3 & 1).mapping

    def test_xor(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 ^ d3).mapping == {0: 3, 3: 2, 2: 2, 1: 2}
        assert (d3 ^ 1).mapping == {0: 1, 3: 1, 2: 1}
        assert (1 ^ d3).mapping == (d3 ^ 1).mapping

    def test_or(self) -> None:
        d3 = PMF((1, 2, 3))
        assert (d3 | d3).mapping == {1: 1, 3: 7, 2: 1}
        assert (d3 | 1).mapping == {1: 1, 3: 2}
        assert (1 | d3).mapping == (d3 | 1).mapping


class TestPMFCollection:
    def test_contains(self) -> None:
        pmf = PMF((1, 2, 3))
        assert 1 in pmf
        assert 2 in pmf
        assert 3 in pmf
        assert 0 not in pmf
        assert "cat" not in pmf  # type: ignore
        assert None not in pmf

    def test_iter(self) -> None:
        pmf = PMF((1, 2, 3))
        items = iter(pmf)
        assert next(items, None) == 1
        assert next(items, None) == 2
        assert next(items, None) == 3
        assert next(items, None) is None

    def test_reversed(self) -> None:
        pmf = PMF((1, 2, 3))
        items = reversed(pmf)
        assert next(items, None) == 3
        assert next(items, None) == 2
        assert next(items, None) == 1
        assert next(items, None) is None

    def test_len(self) -> None:
        pmf = PMF((1, 2, 3))
        assert len(pmf) == 3
        pmf = PMF()
        assert len(pmf) == 0

    def test_index(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        pmf = PMF(pets)
        for ev in pets:
            assert pmf.index(ev) == pets.index(ev)

    def test_index_from_self(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        pmf1 = PMF(pets)
        pmf2 = PMF.from_self(pmf1)
        for ev in pets:
            assert pmf1.index(ev) == pets.index(ev)
            assert pmf2.index(ev) == pets.index(ev)

    def test_index_from_pairs(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        pairs = tuple((ev, i) for i, ev in enumerate(pets))
        pmf = PMF.from_pairs(pairs)
        for ev, i in pairs:
            assert pmf.index(ev) == i

    def test_index_from_iterable(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        pmf = PMF.from_iterable(pets)
        for ev in pets:
            assert pmf.index(ev) == pets.index(ev)

    def test_index_errors(self) -> None:
        pets = ("cat", "dog", "bird", "fish", "snake")
        with pytest.raises(ValueError):
            PMF(pets).index("spider")
        with pytest.raises(ValueError):
            PMF(range(1, 7)).index(0)
            PMF(range(1, 7)).index(7)

    def test_hash(self) -> None:
        pmf = PMF((1, 1, 1, 2, 2, 3))
        assert hash(pmf) == hash(pmf.pairs)


class TestMultisetMath:
    def test_multiset_comb(self) -> None:
        assert multiset_comb(3, 5) == math.comb(7, 5)
        assert multiset_comb(6, 6) == math.comb(11, 6)

    def test_multiset_perm(self) -> None:
        items = (3, 3, 5, 6, 9)
        perm = math.factorial(sum(items))
        for item in items:
            perm //= math.factorial(item)
        assert multiset_perm(items) == perm


class TestQuantileName:
    def test_quantile_name(self) -> None:
        assert quantile_name(0) == "0-quantile"
        assert quantile_name(2) == "median"
        assert quantile_name(4) == "quartile"
        assert quantile_name(5) == "quintile"
        assert quantile_name(10) == "decile"
        assert quantile_name(100) == "centile"

    def test_quantile_name_plural(self) -> None:
        assert quantile_name(0, plural=True) == "0-quantiles"
        assert quantile_name(2, plural=True) == "medians"
        assert quantile_name(4, plural=True) == "quartiles"
        assert quantile_name(5, plural=True) == "quintiles"
        assert quantile_name(10, plural=True) == "deciles"
        assert quantile_name(100, plural=True) == "centiles"
        assert quantile_name(2, kind=QK.GROUP, plural=True) == "halves"
        assert quantile_name(2, kind=QK.FRACTION, plural=True) == "halves"

    def test_quantile_name_group(self) -> None:
        assert quantile_name(0, kind=QK.GROUP) == "0-quantile"
        assert quantile_name(2, kind=QK.GROUP) == "half"
        assert quantile_name(4, kind=QK.GROUP) == "quartile"
        assert quantile_name(5, kind=QK.GROUP) == "quintile"
        assert quantile_name(10, kind=QK.GROUP) == "decile"
        assert quantile_name(100, kind=QK.GROUP) == "centile"

    def test_quantile_name_fraction(self) -> None:
        assert quantile_name(0, kind=QK.FRACTION) == "0-quantile"
        assert quantile_name(2, kind=QK.FRACTION) == "half"
        assert quantile_name(4, kind=QK.FRACTION) == "quarter"
        assert quantile_name(5, kind=QK.FRACTION) == "fifth"
        assert quantile_name(10, kind=QK.FRACTION) == "tenth"
        assert quantile_name(100, kind=QK.FRACTION) == "hundredth"
