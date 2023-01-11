"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "Operator",
    "PMF",
    "Probability",
    "Weight",
    "multiset_comb",
    "multiset_perm",
]

import functools
import itertools
import math
import numbers
import operator
import typing
from collections import Counter
from collections.abc import (
    Callable,
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
from typing import Any, cast, Self, TypeAlias, TypeVar

# Type variables.
ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.

# Type aliases.
Operator: TypeAlias = Callable[..., Any]
Probability: TypeAlias = Fraction
Weight: TypeAlias = int


class PMF(Collection[ET_co]):
    """Generic base class for probability mass functions.

    A probability mass function (PMF) associates probabilities with
    events in a discrete probability space.

    Probability spaces model randomness as a triplet (Ω, Σ, P):

    * Ω, the sample space, which is the set of all outcomes.
    * Σ, the event space, which groups outcomes into sets for analysis.
    * P, a function that associates a probability with every event.

    Outcomes are the specific results of a random process.  For example,
    given a deck of 52 playing cards, a single draw has 52 outcomes.

    Events are interesting sets of outcomes.  They can comprise one
    outcome ("draw the queen of hearts") or many ("draw any odd-numbered
    card").  The singletons are called elementary events, and they're
    often used interchangeably with the corresponding outcome.

    Probability functions assign probabilities to events, and a PMF
    typically maps all of the elementary events in the event space.  For
    example, the PMF for a coin flip is {heads: 1/2, tails: 1/2}.

    This class represents probability as int weights instead of floats
    or Fractions.  To convert to the usual [0, 1] range for probability,
    divide by the total property, or use the "mass" methods instead of
    "weight" methods.
    """

    # TODO: event type check and conversion
    __weights: Mapping[ET_co, Weight]
    __total: Weight

    def __init__(
        self,
        events: Mapping[ET_co, Weight] | Iterable[ET_co] = (),
        /,
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize PMF object."""
        # Convert input mapping or iterable to a list of pairs.
        pairs: Iterable[tuple[ET_co, Weight]]
        match events:
            case PMF() if type(events) is type(self):
                # Copy another PMF of the same type.
                if not normalize:
                    self.__weights = events.__weights
                    self.__total = events.__total
                    return
                pairs = events.pairs
            case Mapping():
                pairs = cast(ItemsView[ET_co, Weight], events.items())
            case Iterable():
                pairs = ((event, 1) for event in events)
            case _:
                raise TypeError(f"not iterable: {type(events).__name__!r}")
        # Collect event weights.
        weights: dict[ET_co, Weight] = {}
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

    @classmethod
    def convert(cls, other: Any, /) -> Self:
        """Convert an object to a PMF."""
        if isinstance(other, PMF):
            # If the object is already a PMF, convert it to the same
            # subtype (if necessary) and then return it.
            pmf: PMF[Any] = other
            return pmf if type(pmf) is cls else cls(pmf)
        # Otherwise, convert the object to a single-event PMF.
        weights: dict[ET_co, Weight] = {other: 1}
        return cls(weights)

    @functools.cached_property
    def domain(self) -> Sequence[ET_co]:
        """Return all events defined for the PMF."""
        return tuple(self.mapping)

    @functools.cached_property
    def support(self) -> Sequence[ET_co]:
        """Return all events with non-zero probability."""
        return tuple(v for v, p in self.mapping.items() if p)

    @functools.cached_property
    def weights(self) -> Sequence[Weight]:
        """Return all event weights defined for the PMF."""
        return tuple(self.mapping.values())

    @functools.cached_property
    def pairs(self) -> Sequence[tuple[ET_co, Weight]]:  # TODO: rename?
        """Return all of the (event, weight) pairs."""
        return tuple(self.mapping.items())

    @property
    def mapping(self) -> Mapping[ET_co, Weight]:
        """Provide read-only access to the probability mapping."""
        return self.__weights

    @property
    def total(self) -> Weight:
        """Provide read-only access to the total probability."""
        return self.__total

    @functools.cached_property
    def image(self) -> Sequence[Probability]:
        """Return all event probabilities."""
        return tuple(Fraction(w, self.total) for w in self.weights)

    @functools.cached_property
    def graph(self) -> Sequence[tuple[ET_co, Probability]]:
        """Return all of the (event, probability) pairs."""
        return tuple((v, Fraction(w, self.total)) for v, w in self.mapping.items())

    def probability(self, event: Any, /) -> Probability:
        """Return the probability of a given event."""
        weight = self.weight(event)
        return Fraction(weight, self.total or 1)

    __call__ = probability

    def weight(self, event: Any, /) -> Weight:
        """Return the probability weight of a given event."""
        try:
            weight = self.mapping.get(event, 0)
        except TypeError:  # not Hashable
            weight = 0
        return weight

    def normalized(self) -> Self:
        """Return an equivalent object with minimum integer weights."""
        return type(self)(self, normalize=True)

    def copy(self) -> Self:
        """Create a shallow copy."""
        return type(self)(self, normalize=False)

    def plot(self) -> None:
        """Display the PMF with matplotlib."""
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover
            return
        fig, ax = plt.subplots()
        domain = self.domain
        image = self.image
        ax.bar(domain, image)
        # ax.ylabel("Probability")
        # ax.title("Hits")
        plt.show()
        # if sys.__stdout__.isatty():  # pragma: no cover
        #     plt.ion()
        #     plt.show(block=True)

    def tabulate(
        self,
        __spec: str = "",
        /,
        *,
        align: bool = True,
        separator: str | None = None,
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

    @functools.lru_cache
    def combinations(self, n: int, /) -> Iterable[Sequence[ET_co]]:
        """Generate all distinct combinations of N outcomes."""
        if n < 0:
            raise ValueError("combinations must be non-negative")
        combos = tuple(itertools.combinations_with_replacement(self.domain, n))
        return combos

    @functools.lru_cache
    def combination_weights(self, n: int) -> Mapping[Sequence[ET_co], Weight]:
        """Generate a weight mapping for the combinations method."""
        weights: dict[Sequence[ET_co], Weight] = {}
        for combo in self.combinations(n):
            counter = Counter(combo)
            counts = tuple(sorted(counter.values()))
            cperms = multiset_perm(counts)
            cweight = math.prod(self.weight(v) for v in counter.elements())
            weights[combo] = cperms * cweight
        return MappingProxyType(weights)

    def map(self, f: Operator, /) -> Self:
        """Map events through a callable."""
        weights: dict[ET_co, Weight] = {}
        events: list[tuple[Any, Weight]] = []
        totals: list[int] = []
        # Get the mapped values.
        for ev, wt in self.pairs:
            mv = f(ev)
            events.append((mv, wt))
            if isinstance(mv, PMF):
                totals.append(mv.total)
        # Find a common weight multiplier for PMF results.
        share = math.lcm(*totals)
        # Determine new weights for everything.
        for mv, wt in events:
            if not isinstance(mv, PMF):
                weights[mv] = weights.setdefault(mv, 0) + wt * share
                continue
            # Unpack PMFs.
            mv = cast(Self, mv)  # assume a compatible PMF
            pshare = share // mv.total
            for pv, pwt in mv.pairs:
                weights[pv] = weights.setdefault(pv, 0) + wt * pwt * pshare
        return type(self)(weights)

    @functools.lru_cache
    def times(self, n: int, op: Operator = operator.add, /) -> Self:
        """Fold a PMF with itself to evaluate an n-ary operation."""
        if n < 1:
            raise ValueError("repetitions must be strictly positive")
        if n == 1:
            return self
        result = self.binary_operator(self, op)
        for _ in range(2, n):
            result = result.binary_operator(self, op)
        return result

    @functools.lru_cache
    def rtimes(self, n: int, op: Operator = operator.add, /) -> Self:
        """Right-associative version of the times method."""
        if n < 1:
            raise ValueError("repetitions must be strictly positive")
        if n == 1:
            return self
        result = self.binary_operator(self, op)
        for _ in range(2, n):
            result = self.binary_operator(result, op)
        return result

    def unary_operator(self, op: Operator, /, *args: Any, **kwargs: Any) -> Self:
        """Compute a unary operator over a PMFs."""
        weights: dict[ET_co, Weight] = {}
        for ev, wt in self.pairs:
            ev = op(ev, *args, **kwargs)
            weights[ev] = weights.setdefault(ev, 0) + wt
        return type(self)(weights)

    def __neg__(self) -> Self:
        """Compute -self."""
        return self.unary_operator(operator.neg)

    def __pos__(self) -> Self:
        """Compute +self."""
        return self.unary_operator(operator.pos)

    def __abs__(self) -> Self:
        """Compute abs(self)."""
        return self.unary_operator(operator.abs)

    def __invert__(self) -> Self:
        """Compute ~self."""
        return self.unary_operator(operator.invert)

    def __round__(self, ndigits: int | None = None) -> Self:
        """Compute round(self)."""
        return self.unary_operator(round, ndigits=ndigits)

    def __trunc__(self) -> Self:
        """Compute math.trunc(self)."""
        return self.unary_operator(math.trunc)

    def __floor__(self) -> Self:
        """Compute math.floor(self)."""
        return self.unary_operator(math.floor)

    def __ceil__(self) -> Self:
        """Compute math.ceil(self)."""
        return self.unary_operator(math.ceil)

    def binary_operator(
        self, other: Self, op: Operator, /, *args: Any, **kwargs: Any
    ) -> Self:
        """Compute a binary operator between two PMFs."""
        weights: dict[ET_co, Weight] = {}
        for ev2, wt2 in other.pairs:
            for ev1, wt1 in self.pairs:
                ev = op(ev1, ev2, *args, **kwargs)
                weights[ev] = weights.setdefault(ev, 0) + wt1 * wt2
        return type(self)(weights)

    def __matmul__(self, other: Any) -> Self:
        """Compute self @ other."""
        other = self.convert(other)
        return self.map(other.times)

    def __rmatmul__(self, other: Any) -> Self:
        """Compute other @ self."""
        if isinstance(other, typing.SupportsInt):
            return self.times(int(other))
        return self.convert(other).__matmul__(self)

    def __add__(self, other: Any) -> Self:
        """Compute self + other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.add)

    def __radd__(self, other: Any) -> Self:
        """Compute other + self."""
        return self.convert(other).binary_operator(self, operator.add)

    def __sub__(self, other: Any) -> Self:
        """Compute self - other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.sub)

    def __rsub__(self, other: Any) -> Self:
        """Compute other - self."""
        return self.convert(other).binary_operator(self, operator.sub)

    def __mul__(self, other: Any) -> Self:
        """Compute self * other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.mul)

    def __rmul__(self, other: Any) -> Self:
        """Compute other * self."""
        return self.convert(other).binary_operator(self, operator.mul)

    def __truediv__(self, other: Any) -> Self:
        """Compute self / other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Self:
        """Compute other / self."""
        return self.convert(other).binary_operator(self, operator.truediv)

    def __floordiv__(self, other: Any) -> Self:
        """Compute self // other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> Self:
        """Compute other // self."""
        return self.convert(other).binary_operator(self, operator.floordiv)

    def __mod__(self, other: Any) -> Self:
        """Compute self % other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.mod)

    def __rmod__(self, other: Any) -> Self:
        """Compute other % self."""
        return self.convert(other).binary_operator(self, operator.mod)

    def __pow__(self, other: Any) -> Self:
        """Compute self ** other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.pow)

    def __rpow__(self, other: Any) -> Self:
        """Compute other ** self."""
        return self.convert(other).binary_operator(self, operator.pow)

    def __lshift__(self, other: Any) -> Self:
        """Compute self << other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.lshift)

    def __rlshift__(self, other: Any) -> Self:
        """Compute other << self."""
        return self.convert(other).binary_operator(self, operator.lshift)

    def __rshift__(self, other: Any) -> Self:
        """Compute self >> other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.rshift)

    def __rrshift__(self, other: Any) -> Self:
        """Compute other >> self."""
        return self.convert(other).binary_operator(self, operator.rshift)

    def __and__(self, other: Any) -> Self:
        """Compute self & other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.and_)

    def __rand__(self, other: Any) -> Self:
        """Compute other & self."""
        return self.convert(other).binary_operator(self, operator.and_)

    def __xor__(self, other: Any) -> Self:
        """Compute self ^ other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.xor)

    def __rxor__(self, other: Any) -> Self:
        """Compute other ^ self."""
        return self.convert(other).binary_operator(self, operator.xor)

    def __or__(self, other: Any) -> Self:
        """Compute self | other."""
        other = self.convert(other)
        return self.binary_operator(other, operator.or_)

    def __ror__(self, other: Any) -> Self:
        """Compute other | self."""
        return self.convert(other).binary_operator(self, operator.or_)

    def __format__(self, spec: str) -> str:
        """Format the PMF according to the format spec."""
        rows = self.tabulate(spec, align=False)
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

    def __contains__(self, event: Any) -> bool:
        """Test object for membership in the event domain."""
        return bool(self.mapping.get(event, 0))

    def __iter__(self) -> Iterator[ET_co]:
        """Iterate over the event domain."""
        return iter(self.domain)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.mapping)

    @functools.cached_property
    def _hash(self) -> int:
        return hash(self.pairs)

    def __hash__(self) -> int:
        """Calculate a hash value from the (event, weight) pairs."""
        return self._hash


@functools.cache
def multiset_comb(n: int, k: int, /) -> int:
    """Count multiset combinations for k items chosen from n options."""
    return math.comb(n + k - 1, k)


@functools.cache
def multiset_perm(items: Iterable[int], /) -> int:
    """Count multiset permutations for item counts in k[n].

    The iterable parameter provides the sizes of the multiset's
    equivalence classes.  For example, the multiset AAABBCC has
    three equivalence classes of size (3, 2, 2).

    The general formula is N! / ∏(k[n]!) where:
    - N    is the total number of items ∑k[n], and
    - k[n] is the number of items in each equivalence class.
    """
    k = sorted(tuple(items))
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
