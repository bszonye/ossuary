"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "PMF",
    "multiset_comb",
    "multiset_perm",
]

import functools
import math
import numbers
from collections.abc import (
    Collection,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast, Optional, Self, TypeAlias, TypeVar

# PMF input types.
# TODO: Rename these to EventT and WeightT?
# TODO: Export the type vars in __all__?
ET_co = TypeVar("ET_co", covariant=True)  # Event type.
WT: TypeAlias = int  # Weight type.
PT: TypeAlias = Fraction  # Probability type.


class PMF(Collection[ET_co]):
    """Generic base class for probability mass functions.

    A probability mass function (PMF) associates probabilities with
    events in a discrete probability space.

    Probability spaces model randomness as a triplet (Ω, Σ, P):

    * Ω, the sample space, which is the set of all outcomes.
    * Σ, the event space, which groups sets of outcomes for analysis.
    * P, a function that associates a probability with every event.

    Outcomes are the specific results of a random process, and events
    are sets of outcomes grouped for analysis.  For example, given a
    deck of 52 cards, a single card draw has 52 possible outcomes.
    Events can include one outcome ("draw the queen of hearts") or many
    ("draw any odd-numbered card").  The singletons are called
    elementary events, and they're often used interchangeably with the
    corresponding outcome.

    Probability functions assign probabilities to events, and a PMF
    typically maps all of the elementary events in the event space.  For
    example, the PMF for a coin flip is {heads: 1/2, tails: 1/2}.

    This class represents probability as int weights instead of floats
    or Fractions.  To convert to the usual [0, 1] range for probability,
    divide by the total property, or use the "mass" methods instead of
    "weight" methods.
    """

    __weights: Mapping[ET_co, WT]
    __total: WT

    def __init__(
        self,
        __events: Mapping[ET_co, WT] | Iterable[ET_co] = (),
        /,
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize PMF object."""
        # Convert input mapping or iterable to a list of pairs.
        pairs: Iterable[tuple[ET_co, WT]]
        match __events:
            case PMF() if type(__events) is type(self):
                # Copy another PMF of the same type.
                if not normalize:
                    self.__weights = __events.__weights
                    self.__total = __events.__total
                    return
                pairs = __events.pairs
            case Mapping():
                pairs = cast(ItemsView[ET_co, WT], __events.items())
            case Iterable():
                pairs = ((event, 1) for event in __events)
            case _:
                raise TypeError(f"not iterable: {type(__events).__name__!r}")
        # Collect event weights.
        weights: dict[ET_co, WT] = {}
        total = 0
        for ev, wt in pairs:
            # Check parameter types and values.
            if not isinstance(ev, Hashable):
                raise TypeError(f"unhashable type: {type(ev).__name__!r}")
            if not isinstance(wt, int):  # pyright: ignore
                raise TypeError(f"not a probability weight: {type(wt).__name__!r}")
            if wt < 0:
                raise ValueError(f"not a probability weight: {wt!r} < 0")
            # Record event weight and total weight.
            weights[ev] = weights.setdefault(ev, 0) + wt
            total += wt
        # Optionally, reduce weights by their greatest common divisor.
        if normalize:
            factor = math.gcd(*weights.values())
            if 1 < factor:  # Don't divide by 0 or 1.
                for event in weights.keys():
                    weights[event] //= factor
                total //= factor
        # Initialize attributes.
        self.__weights = MappingProxyType(weights)
        self.__total = total

    @property  # TODO: functools.cached_property?
    def domain(self) -> Sequence[ET_co]:
        """Return all events defined for the PMF."""
        return tuple(self.mapping)

    @property  # TODO: functools.cached_property?
    def support(self) -> Sequence[ET_co]:
        """Return all events with non-zero probability."""
        return tuple(v for v, p in self.mapping.items() if p)

    @property  # TODO: functools.cached_property?
    def weights(self) -> Sequence[WT]:
        """Return all event weights defined for the PMF."""
        return tuple(self.mapping.values())

    @property  # TODO: functools.cached_property?
    def pairs(self) -> Sequence[tuple[ET_co, WT]]:  # TODO: rename?
        """Return all of the (event, weight) pairs."""
        return tuple(self.mapping.items())

    @property  # TODO: functools.cached_property?
    def graph(self) -> Sequence[tuple[ET_co, PT]]:
        """Return all of the (event, probability) pairs."""
        return tuple((v, Fraction(w, self.total)) for v, w in self.mapping.items())

    @property
    def mapping(self) -> Mapping[ET_co, WT]:
        """Provide read-only access to the probability mapping."""
        return self.__weights

    @property
    def total(self) -> WT:
        """Provide read-only access to the total probability."""
        return self.__total

    def probability(self, __event: Any, /) -> PT:
        """Return the probability of a given event."""
        weight = self.weight(__event)
        return Fraction(weight, self.total or 1)

    __call__ = probability

    def weight(self, __event: Any, /) -> WT:
        """Return the probability weight of a given event."""
        try:
            weight = self.mapping.get(__event, 0)
        except TypeError:  # not Hashable
            weight = 0
        return weight

    def normalized(self) -> Self:
        """Return an equivalent object with minimum integer weights."""
        return type(self)(self, normalize=True)

    def copy(self) -> Self:
        """Create a shallow copy."""
        return type(self)(self, normalize=False)

    def tabulate(
        self,
        __spec: str = "",
        /,
        *,
        align: bool = True,
        separator: Optional[str] = None,
    ) -> Sequence[str]:
        """Format PMF as a table."""
        # Validate & initialize parameters.
        if not self:
            return tuple()
        if separator is None:
            separator = "  " if align else ": "
        specs = __spec.split(":", 1)
        vspec: str = specs[0]
        wspec: str = specs[-1]  # Same as vspec if there's no colon.

        # Format columns.
        def column(item: Any, spec: str, width: int = 0, frac: int = 0) -> str:
            """Format an object to spec, with extra width controls."""
            # Fractions don't support format specifiers.
            if spec and isinstance(item, Fraction):
                item = float(item)

            # Simple alignment: numbers right, everything else left.
            # To override, use a format specifier with explicit column
            # widths and alignment/fill options.
            text = format(item, spec)
            return (
                text.rjust(width)
                if isinstance(item, numbers.Number)
                else text.ljust(width)
            )

        def measure(items: Iterable[Any], spec: str) -> int:
            return max(len(column(item, spec)) for item in items) if align else 0

        # Select probability or weight depending on format.  Use integer
        # weights for int formats, fractional probabilities otherwise.
        pfunc = self.weight if wspec and wspec[-1] in "Xbcdnox" else self.probability
        events = self.domain
        # Determine the minimum column & fraction widths.
        vwidth = measure(events, vspec)
        wwidth = measure(map(pfunc, events), wspec)
        # Generate text.
        return tuple(
            separator.join(
                (column(event, vspec, vwidth), column(pfunc(event), wspec, wwidth))
            )
            for event in events
        )

    def __format__(self, __spec: str) -> str:
        """Format the PMF according to the format spec."""
        rows = self.tabulate(__spec, align=False)
        return "{" + ", ".join(rows) + "}"

    def __repr__(self) -> str:
        """Format the PMF for diagnostics."""
        params = (
            repr(dict(self.mapping))
            if self.mapping
            else f"normalize={self.__total!r}"
            if self.__total != 1
            else ""
        )
        return f"{type(self).__name__}({params})"

    def __str__(self) -> str:
        """Format the PMF for printing."""
        return self.__format__("")

    def __contains__(self, __event: Any) -> bool:
        """Test object for membership in the event domain."""
        return bool(self.mapping.get(__event, 0))

    def __iter__(self) -> Iterator[ET_co]:
        """Iterate over the event domain."""
        return iter(self.domain)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.mapping)


@functools.cache
def multiset_comb(__n: int, __k: int, /) -> int:
    """Count multiset combinations for k items chosen from n options."""
    return math.comb(__n + __k - 1, __k)


@functools.cache
def multiset_perm(__iter: Iterable[int], /) -> int:
    """Count multiset permutations for item counts in k[n].

    The iterable parameter provides the sizes of the multiset's
    equivalence classes.  For example, the multiset AAABBCC has
    three equivalence classes of size (3, 2, 2).

    The general formula is N! / ∏(k[n]!) where:
    - N    is the total number of items ∑k[n], and
    - k[n] is the number of items in each equivalence class.
    """
    k = sorted(tuple(__iter))
    n = sum(k)
    if not n:
        return 0
    # Start by dividing N! by the largest k[n]! to keep the running
    # product as small as possible.  Also, use an equivalent permutation
    # to take advantage of optimizations in the math.perm function.
    weight = math.perm(n, n - k[-1])
    # Then divide by the remaining k[n]! factors from large to small.
    for count in reversed(k[:-1]):
        weight //= math.factorial(count)
    return weight
