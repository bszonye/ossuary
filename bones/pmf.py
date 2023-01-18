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

import enum
import functools
import itertools
import math
import numbers
import operator
import types
import typing
from collections import Counter
from collections.abc import (
    Callable,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from fractions import Fraction
from gettext import gettext as _t
from types import MappingProxyType
from typing import (
    Any,
    cast,
    overload,
    Self,
    Sized,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
)

from .color import adjust_lightness, ColorTriplet, interpolate_color

# Type variables.
ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.

# Type aliases.
Auto: TypeAlias = types.EllipsisType
Operator: TypeAlias = Callable[..., Any]
Probability: TypeAlias = Fraction
Weight: TypeAlias = int

# Need explicit unions with the builtin types because mypy does not
# recognize the numeric tower extension mechanism.  The Fraction class
# works fine because subclasses Rational directly.
_Rational: TypeAlias = numbers.Rational | int
_Real: TypeAlias = numbers.Real | float | int


class PMF(Sequence[ET_co]):
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

    This class represents probability internally as integral weights
    instead of floating-point or fractional probabilities.  It offers
    both count-based and probability-based accessors.
    """

    # TODO: event type check and conversion

    __weights: Mapping[ET_co, Weight]
    __total: Weight
    __gcd: Weight
    __index: Mapping[ET_co, int]

    # ==================================================================
    # CONSTRUCTORS

    def __init__(
        self,
        events: Mapping[ET_co, Weight] | Iterable[ET_co] = (),
        /,
        *,
        reverse: bool = False,
        normalize: bool = False,
    ) -> None:
        """Initialize PMF object."""
        self.from_iterable(
            events, instance=self, reverse=reverse, normalize=normalize
        )

    @classmethod
    def from_self(
        cls,
        other: "PMF[ET_co]",
        /,
        *,
        instance: Self | None = None,
        reverse: bool = False,
        normalize: bool = False,
    ) -> Self:
        """Construct a new PMF from an existing one."""
        # Create the instance if it doesn't already exist.
        if instance is None:
            instance = cls.__new__(cls)

        # Rebuild from (event, weight) pairs if type or direction doesn't match.
        if reverse or type(other) is not cls:
            pairs = other.pairs
            return cls.from_pairs(
                pairs, instance=instance, reverse=reverse, normalize=normalize
            )

        if normalize and not other.is_normal():
            gcd = other.gcd
            weights: dict[ET_co, Weight] = {ev: wt // gcd for ev, wt in other.pairs}
            instance.__weights = MappingProxyType(weights)
            instance.__total = other.total // gcd
            instance.__gcd = 1
        else:
            instance.__weights = other.__weights
            instance.__total = other.__total
            instance.__gcd = other.__gcd
        instance.__index = other.__index
        return instance

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[ET_co, Weight]],
        /,
        *,
        instance: Self | None = None,
        reverse: bool = False,
        normalize: bool = False,
    ) -> Self:
        """Construct a new PMF from (event, weight) pairs."""
        # Create the instance if it doesn't already exist.
        if instance is None:
            instance = cls.__new__(cls)
        # Reverse as needed.
        if reverse:
            pairs = reversed(tuple(pairs))
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
        gcd = math.gcd(*weights.values())
        if normalize and 1 < gcd:
            for event in weights.keys():
                weights[event] //= gcd
            total //= gcd
            gcd = 1
        # Initialize attributes.
        instance.__weights = MappingProxyType(weights)
        instance.__total = total
        instance.__gcd = gcd
        instance.__index = MappingProxyType(
            {ev: i for i, ev in enumerate(instance.domain)}
        )
        return instance

    @classmethod
    def from_iterable(
        cls,
        events: Iterable[ET_co],
        /,
        *,
        instance: Self | None = None,
        reverse: bool = False,
        normalize: bool = False,
    ) -> Self:
        """Create a new PMF from an iterable."""
        # Create the instance if it doesn't already exist.
        if instance is None:
            instance = cls.__new__(cls)
        pairs: Iterable[tuple[ET_co, Weight]]
        match events:
            case PMF():
                return instance.from_self(
                    events, instance=instance, reverse=reverse, normalize=normalize
                )
            case Mapping():
                # The from_pairs constructor will check Weight types.
                pairs = cast(ItemsView[ET_co, Weight], events.items())
            case Iterable():
                pairs = ((event, 1) for event in events)
            case _:
                raise TypeError(f"not iterable: {type(events).__name__!r}")
        # Finish initializion from the (event, weight) pairs.
        return instance.from_pairs(
            pairs, instance=instance, reverse=reverse, normalize=normalize
        )

    @classmethod
    def convert(cls, other: Any, /, normalize: bool = False) -> Self:
        """Convert an object to a PMF."""
        # Return the other object directly if it's the same type of PMF.
        # Use the from_self constructor for other subtypes.
        if isinstance(other, PMF):
            pmf: PMF[Any] = other
            return pmf if type(pmf) is cls else cls.from_self(pmf)
        # Otherwise, convert the object to a single-event PMF.
        # There's no implicit conversion of weight mappings or pairs!
        # Use from_iterable or from_pairs for explicit conversion.
        pair: tuple[ET_co, Weight] = (other, 1)
        return cls.from_pairs((pair,))

    # ==================================================================
    # PROPERTIES AND ACCESSORS

    @property
    def mapping(self) -> Mapping[ET_co, Weight]:
        """Provide read-only access to the probability mapping."""
        return self.__weights

    @property
    def total(self) -> Weight:
        """Provide read-only access to the total probability."""
        return self.__total

    @property
    def gcd(self) -> Weight:
        """Provide read-only access to the GCD of all event weights."""
        return self.__gcd

    @functools.cached_property
    def domain(self) -> tuple[ET_co, ...]:
        """Return all events defined for the PMF."""
        return tuple(self.mapping)

    @functools.cached_property
    def support(self) -> tuple[ET_co, ...]:
        """Return all events with non-zero probability."""
        return tuple(ev for ev, wt in self.mapping.items() if wt)

    @functools.cached_property
    def zeroes(self) -> tuple[ET_co, ...]:
        """Return all events with zero probability."""
        return tuple(ev for ev, wt in self.mapping.items() if not wt)

    @functools.cached_property
    def weights(self) -> tuple[Weight, ...]:
        """Return all event weights defined for the PMF."""
        return tuple(self.mapping.values())

    @functools.cached_property
    def sum_weights(self) -> tuple[Weight, ...]:
        """Return the cumulative (<=) weights for all events."""
        return tuple(itertools.accumulate(self.weights))

    @functools.cached_property
    def tail_weights(self) -> tuple[Weight, ...]:
        """Return the tail (not <=) weights for all events."""
        return tuple(self.total - wt for wt in self.sum_weights)

    @functools.cached_property
    def pairs(self) -> tuple[tuple[ET_co, Weight], ...]:
        """Return all of the (event, weight) pairs."""
        return tuple(self.mapping.items())

    @functools.cached_property
    def image(self) -> tuple[Probability, ...]:
        """Return all event probabilities."""
        return tuple(Fraction(w, self.total) for w in self.weights)

    @functools.cached_property
    def graph(self) -> tuple[tuple[ET_co, Probability], ...]:
        """Return all of the (event, probability) pairs."""
        return tuple((v, Fraction(w, self.total)) for v, w in self.mapping.items())

    def probability(self, event: Any, /) -> Probability:
        """Return the probability of a given event."""
        # TODO: accept multiple arguments?
        weight = self.weight(event)
        return Fraction(weight, self.total or 1)

    __call__ = probability

    def weight(self, event: Any, /) -> Weight:
        """Return the probability weight of a given event."""
        # TODO: accept multiple arguments?
        try:
            weight = self.mapping.get(event, 0)
        except TypeError:  # not Hashable
            weight = 0
        return weight

    # ==================================================================
    # COPYING AND NORMALIZATION

    def copy(self, normalize: bool = False) -> Self:
        """Create a shallow copy."""
        return self.from_self(self, normalize=normalize)

    def is_normal(self) -> bool:
        """Return true if the PMF weights are irreducible."""
        return self.__gcd <= 1

    def normalized(self) -> Self:
        """Create a copy with all weights reduced to lowest terms."""
        return self if self.is_normal() else self.copy(normalize=True)

    def is_sorted(
        self,
        *,
        key: Operator | None = None,
        reverse: bool = False,
    ) -> bool:
        """Return true if the PMF domain is in sorted order."""
        if len(self) < 2:
            return True

        # Build a sequence in the right direction with key applied.
        items: Iterable[Any]  # assume event or key type is sortable
        items = reversed(self.domain) if reverse else self.domain
        if key is not None:
            items = map(key, items)
        return all(a <= b for a, b in itertools.pairwise(items))

    def sorted(
        self,
        *,
        key: Operator | None = None,
        reverse: bool = False,
        normalize: bool = False,
    ) -> Self:
        """Create a sorted copy."""
        key_or_identity = key or (lambda x: x)

        def pairkey(pair: tuple[ET_co, Weight]) -> Any:
            return key_or_identity(pair[0])

        pairs = sorted(self.pairs, key=pairkey, reverse=reverse)
        return self.from_pairs(pairs, normalize=normalize)

    # ==================================================================
    # STATISTICS

    # TODO: median, mode

    @functools.cached_property
    def _mean_numerator(self: "PMF[_Real]") -> ET_co:
        """Calculate the exact numerator of the mean."""
        return cast(ET_co, sum(map(operator.mul, self.domain, self.weights)))

    @functools.cached_property
    def exact_mean(self: "PMF[_Rational]") -> Fraction:
        """Calculate the PMF mean as a fraction."""
        return Fraction(self._mean_numerator, self.total)

    @functools.cached_property
    def mean(self: "PMF[_Real]") -> float:
        """Calculate the PMF mean as a float."""
        return float(self._mean_numerator / self.total)

    @functools.cached_property
    def _variance_squares(self: "PMF[_Real]") -> ET_co:
        """Calculate the sum of squares used in calculating variance."""
        sum_of_squares = sum(map(lambda v, w: v * v * w, self.domain, self.weights))
        return cast(ET_co, sum_of_squares)

    @functools.cached_property
    def exact_variance(self: "PMF[_Rational]") -> Fraction:
        """Calculate the PMF variance as a fraction."""
        mean_of_squares = Fraction(self._variance_squares, self.total)
        square_of_mean = self.exact_mean * self.exact_mean
        return mean_of_squares - square_of_mean

    @functools.cached_property
    def variance(self: "PMF[_Real]") -> float:
        """Calculate the PMF variance as a float."""
        mean_of_squares = float(self._variance_squares / self.total)
        square_of_mean = self.mean * self.mean
        return mean_of_squares - square_of_mean

    @functools.cached_property
    def standard_deviation(self: "PMF[_Real]") -> float:
        """Calculate the PMF standard distribtion as a float."""
        return math.sqrt(self.variance)

    def population(self) -> Iterator[ET_co]:
        """Iterate over all events, repeated by weight."""
        return itertools.chain.from_iterable(
            itertools.starmap(itertools.repeat, self.pairs)
        )

    def quantiles(self, n: int | Auto = Ellipsis, /) -> Sequence[tuple[ET_co, ...]]:
        """Partition the domain into equally likely groups.

        Quantiles divide a probability distribution into continuous,
        equal intervals.  The term can refer either to the cut points or
        the equal groups; this method returns the groups.  If there is
        an exact median and an even number of groups, this method places
        the median into the upper group.
        """
        size = len(self)
        n = (
            n
            if isinstance(n, int)
            else 1
            if size < 4
            else 4
            if size in (4, 6, 8, 12, 16)
            else 5
            if size < 18
            else 10
        )

        if n < 0:
            raise ValueError("size must be non-negative")
        if n < 2:  # all in one group
            return (self.domain,)

        groups: list[list[ET_co]] = [[] for _ in range(n)]

        # Scale sizes to the least common multiple to avoid rounding.
        total = math.lcm(2 * self.total, n)
        scale = total // self.total
        bucket = total // n

        pairs = iter(self.pairs)
        acc = 0
        try:
            ev, wt = next(pairs)
            wt *= scale
            for i in range(n):
                limit = bucket * (i + 1)
                while (midpoint := acc + wt // 2) <= limit:
                    # If a quantile falls exactly in the middle of an
                    # event, round toward the center for symmetry.
                    if midpoint == limit and 2 * limit <= total:
                        break
                    acc += wt
                    groups[i].append(ev)
                    ev, wt = next(pairs)
                    wt *= scale
        except StopIteration:
            pass
        assert acc == total

        return tuple(tuple(group) for group in groups)

    # ==================================================================
    # TABULATON AND OUTPUT

    def plot_colors(
        self,
        *,
        q: None | int | Auto = Ellipsis,
    ) -> Iterable[ColorTriplet]:
        """Determine plot colors for all event probabilities."""
        return ()  # TODO

    def plot(
        self,
        *,
        q: None | int | Auto = Ellipsis,
        vformat: str = "",
        precision: int = 2,
        window_title: str = "bones",
    ) -> None:
        """Display the PMF with matplotlib."""
        try:
            from matplotlib import pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:  # pragma: no cover
            return

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(window_title)

        # To temporarily change plot style, use this context manager.
        # with plt.rc_context({"axes.labelsize": 20}):
        domain = self.domain
        image = tuple(float(self(v)) for v in domain)
        n = len(domain)

        precision = max(0, precision)
        vlabels = tuple(format(v, vformat) for v in domain)
        plabels = tuple(f"{100*p:.{precision}f}" for p in image)
        legend: list[str] = []

        # Group events into quantiles.
        quantiles = self.quantiles(q or 0)
        nq = len(quantiles)
        color: tuple[ColorTriplet, ...]
        if 2 <= nq:  # Set up quantile colors.
            color = tuple()
            for i in range(nq):
                # Color quantiles from blue to red to green.
                qmax = nq - 1
                hues = min(max(180, 30 * qmax), 285)
                hmin = (0 - hues / 2) / 360
                hmax = (0 + hues / 2) / 360
                qcolor = interpolate_color(
                    i, tmax=qmax, hmin=hmin, hmax=hmax, lmin=0.15, lmax=0.25
                )
                color = color + (qcolor,) * len(quantiles[i])
                # Label quantiles from 1/N to N/N.
                width = len(str(nq))
                fill = "\u2007"  # U+2007 FIGURE SPACE, &numsp;
                label = f"{1+i:{fill}>{width}d}/{nq}"
                legend.append(Patch(color=qcolor, label=label))
        else:
            # Color probabilities from violet to red.
            color = tuple(
                interpolate_color(i, tmin=min(image), tmax=max(image)) for i in image
            )
        edge = tuple(adjust_lightness(0.65, c) for c in color)

        chart = ax.bar(
            x=range(n),
            height=image,
            tick_label=vlabels,
            color=color,
            edgecolor=edge,
            # hatch="/",
        )
        ax.set_xlabel("Events")  # TODO: customizable title
        ax.set_ylabel("Probability")
        ax.bar_label(chart, labels=plabels, padding=1, fontsize="x-small")
        if legend:
            ax.legend(
                title=quantile_name(len(quantiles)),
                title_fontsize="small",
                handles=legend,
                fontsize="x-small",
            )

        with plt.rc_context({"hatch.linewidth": 8.5}):
            plt.show()

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

    def __format__(self, spec: str) -> str:
        """Format the PMF according to the format spec."""
        rows = self.tabulate(spec, align=False)
        return "{" + ", ".join(rows) + "}"

    def __repr__(self) -> str:
        """Format the PMF for diagnostics."""
        parameters: list[str] = []
        if self.total:
            parameters.append(repr(dict(self.mapping)))
        if 1 < self.gcd:
            parameters.append("normalize=False")
        parameter_list = ", ".join(parameters)
        return f"{type(self).__name__}({parameter_list})"

    def __str__(self) -> str:
        """Format the PMF for printing."""
        return self.__format__("")

    # ==================================================================
    # COMBINATORICS

    @functools.lru_cache
    def combinations(self, n: int, /) -> Iterable[tuple[ET_co, ...]]:
        """Generate all distinct combinations of N outcomes."""
        if n < 0:
            raise ValueError("combinations must be non-negative")
        combos = tuple(itertools.combinations_with_replacement(self.domain, n))
        return combos

    @functools.lru_cache
    def combination_weights(self, n: int) -> Mapping[tuple[ET_co, ...], Weight]:
        """Generate a weight mapping for the combinations method."""
        weights: dict[tuple[ET_co, ...], Weight] = {}
        for combo in self.combinations(n):
            counter = Counter(combo)
            counts = tuple(sorted(counter.values()))
            cperms = multiset_perm(counts)
            cweight = math.prod(self.weight(v) for v in counter.elements())
            weights[combo] = cperms * cweight
        return MappingProxyType(weights)

    # ==================================================================
    # HIGHER-ORDER METHODS

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
        return self.from_pairs(weights.items())

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

    # ==================================================================
    # UNARY OPERATORS

    def unary_operator(self, op: Operator, /, *args: Any, **kwargs: Any) -> Self:
        """Compute a unary operator over a PMFs."""
        weights: dict[ET_co, Weight] = {}
        for ev, wt in self.pairs:
            ev = op(ev, *args, **kwargs)
            weights[ev] = weights.setdefault(ev, 0) + wt
        return self.from_pairs(weights.items())

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

    # ==================================================================
    # BINARY OPERATORS

    def binary_operator(
        self, other: Self, op: Operator, /, *args: Any, **kwargs: Any
    ) -> Self:
        """Compute a binary operator between two PMFs."""
        weights: dict[ET_co, Weight] = {}
        for ev2, wt2 in other.pairs:
            for ev1, wt1 in self.pairs:
                ev = op(ev1, ev2, *args, **kwargs)
                weights[ev] = weights.setdefault(ev, 0) + wt1 * wt2
        return self.from_pairs(weights.items())

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

    # ==================================================================
    # SEQUENCE METHODS

    def __contains__(self, event: Any) -> bool:
        """Test object for membership in the event domain."""
        return event in self.mapping

    def __iter__(self) -> Iterator[ET_co]:
        """Iterate over the event domain."""
        return iter(self.domain)

    def __reversed__(self) -> Iterator[ET_co]:
        """Iterate over the event domain in reverse."""
        return reversed(self.domain)

    @overload
    def __getitem__(self, i: SupportsIndex) -> ET_co:  # noqa: D105
        ...

    @overload
    def __getitem__(self, i: slice) -> tuple[ET_co, ...]:  # noqa: D105
        ...

    def __getitem__(self, i: SupportsIndex | slice) -> ET_co | tuple[ET_co, ...]:
        """Get the indexed item or slice from the event domain."""
        return self.domain[i]

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.mapping)

    def index(self, event: Any, /) -> int:  # type: ignore[override]
        """Return the event's position in the domain."""
        try:
            return self.__index[event]
        except KeyError:
            raise ValueError(f"{event!r} not in {type(self).__name__}") from None

    def count(self, event: Any, /) -> int:  # pyright: ignore
        """Return the event's probability weight as its count."""
        return self.weight(event)

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


quantile_names = (
    {
        0: (_t("{}-quartile"), _t("{}-quartiles")),
        2: (_t("median"), _t("medians")),
        3: (_t("tertile"), _t("tertiles")),
        4: (_t("quartile"), _t("quartiles")),
        5: (_t("quintile"), _t("quintiles")),
        6: (_t("sextile"), _t("sextiles")),
        7: (_t("septile"), _t("septiles")),
        8: (_t("octile"), _t("octiles")),
        10: (_t("decile"), _t("deciles")),
        16: (_t("hexadecile"), _t("hexadeciles")),
        20: (_t("ventile"), _t("ventiles")),
        100: (_t("centile"), _t("centiles")),
    },
    {
        0: (_t("{}-quartile"), _t("{}-quartiles")),
        1: (_t("whole"), _t("wholes")),
        2: (_t("half"), _t("halves")),
        3: (_t("tertile"), _t("tertiles")),
        4: (_t("quartile"), _t("quartiles")),
        5: (_t("quintile"), _t("quintiles")),
        6: (_t("sextile"), _t("sextiles")),
        7: (_t("septile"), _t("septiles")),
        8: (_t("octile"), _t("octiles")),
        10: (_t("decile"), _t("deciles")),
        16: (_t("hexadecile"), _t("hexadeciles")),
        20: (_t("ventile"), _t("ventiles")),
        100: (_t("centile"), _t("centiles")),
    },
    {
        0: (_t("{}-quartile"), _t("{}-quartiles")),
        1: (_t("whole"), _t("wholes")),
        2: (_t("half"), _t("halves")),
        3: (_t("third"), _t("thirds")),
        4: (_t("quarter"), _t("quarters")),
        5: (_t("fifth"), _t("fifths")),
        6: (_t("sixth"), _t("sixths")),
        7: (_t("seventh"), _t("sevenths")),
        8: (_t("eighth"), _t("eighths")),
        9: (_t("ninth"), _t("ninths")),
        10: (_t("tenth"), _t("tenths")),
        12: (_t("twelfth"), _t("twelfths")),
        16: (_t("sixteenth"), _t("sixteenths")),
        20: (_t("twentieth"), _t("twentieths")),
        100: (_t("hundredth"), _t("hundredths")),
    },
)


class QK(enum.IntEnum):
    CUT = 0
    GROUP = 1
    FRACTION = 2


@functools.cache
def quantile_name(
    quantile: int | Sized, /, *, kind: QK = QK.GROUP, plural: bool = True
) -> str:
    """Return the name for a quantile of given size."""
    size = int(quantile) if isinstance(quantile, SupportsInt) else len(quantile)
    default = quantile_names[kind][0]
    name = quantile_names[kind].get(size, default)
    return name[int(plural)].format(size)
