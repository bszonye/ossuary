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

import bisect
import enum
import functools
import importlib
import importlib.util
import itertools
import math
import numbers
import operator
import types
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
)
from fractions import Fraction
from gettext import gettext as _t
from types import MappingProxyType
from typing import Any, cast, Self, TypeAlias, TypeVar

from .color import adjust_lightness, color_array, ColorTriplet, interpolate_color

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

    This class represents probability internally as integral weights
    instead of floating-point or fractional probabilities.  It offers
    both count-based and probability-based accessors.
    """

    # TODO: event type check and conversion

    __weights: Mapping[ET_co, Weight]
    __total: Weight
    __gcd: Weight
    __index: Mapping[ET_co, int]
    __quantiles: dict[int, tuple[Self, ...]]

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

        # Rebuild from pairs if type or direction doesn't match.
        if reverse or type(other) is not cls:
            pairs = other.pairs
            return cls.from_pairs(
                pairs, instance=instance, reverse=reverse, normalize=normalize
            )

        # If possible, copy everything, including cached properties.
        if not normalize or other.is_normal():
            instance.__dict__.update(other.__dict__)
            return instance

        gcd = other.gcd
        weights: dict[ET_co, Weight] = {ev: wt // gcd for ev, wt in other.pairs}
        instance.__weights = MappingProxyType(weights)
        instance.__total = other.__total // gcd
        instance.__gcd = 1
        instance.__index = other.__index
        instance.__quantiles = {}
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
        instance.__quantiles = {}
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
    def ranked_pairs(self) -> tuple[tuple[ET_co, Weight], ...]:
        """Return all (event, weight) pairs in order of weight."""
        return tuple(sorted(self.pairs, key=(lambda pair: pair[1])))

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

    def is_ranked(
        self,
        *,
        reverse: bool = False,
    ) -> bool:
        """Return true if the PMF domain is in frequency rank order."""
        if len(self) < 2:
            return True

        # Build a sequence in the right direction with key applied.
        weights: Iterable[Any]  # assume event or key type is sortable
        weights = reversed(self.weights) if reverse else self.weights
        return all(a <= b for a, b in itertools.pairwise(weights))

    def ranked(
        self,
        *,
        reverse: bool = False,
        normalize: bool = False,
    ) -> Self:
        """Create a copy ordered by frequency rank."""
        return self.from_pairs(
            self.ranked_pairs, reverse=reverse, normalize=normalize
        )

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

    @functools.cached_property
    def modal_weight(self) -> Weight:
        """Return the highest weight in the PMF."""
        return max(self.weights) if self.total else 0

    @functools.cached_property
    def multimode(self) -> tuple[ET_co, ...]:
        """Return a tuple of all events with the highest weight."""
        if not len(self):
            return ()
        return tuple(ev for ev, wt in self.pairs if wt == self.modal_weight)

    def population(self) -> Iterator[ET_co]:
        """Iterate over all events, repeated by weight."""
        return itertools.chain.from_iterable(
            itertools.starmap(itertools.repeat, self.pairs)
        )

    @functools.cached_property
    def auto_quantile(self) -> int:
        """Recommend a reasonable quantile size."""
        # TODO: tune this down a bit
        qmax = self.total // (self.modal_weight + 1)
        return (qmax or 1) if qmax < 5 else 5 if qmax < 10 else 10

    def quantile_groups(self, n: int | Auto = Ellipsis, /) -> tuple[Self, ...]:
        """Partition the PMF into equally likely groups."""
        nq = n if isinstance(n, int) else self.auto_quantile
        if nq < 0:
            raise ValueError("size must be non-negative")
        if nq < 2:  # all in one group
            return (self,)

        # Check cache.
        if nq in self.__quantiles:
            return self.__quantiles[nq]

        buckets = tuple(nq * cwt for cwt in self.sum_weights)
        span = self.total
        ql = qr = 0
        groups: list[Self] = []
        for i in range(nq):
            # Cumulative weight at group ends.
            cl = span * i
            cr = span + cl
            # Indexes of events at group ends.
            ql = bisect.bisect_right(buckets, cl, qr)
            qr = bisect.bisect_left(buckets, cr, ql)
            # Collect events & weights in range.
            pairs: list[tuple[ET_co, Weight]] = []
            if ql != qr:
                pairs.append((self.domain[ql], buckets[ql] - cl))
                for j in range(ql + 1, qr):
                    pairs.append((self.domain[j], nq * self.weights[j]))
                pairs.append((self.domain[qr], cr - buckets[qr - 1]))
            else:
                pairs = [(self.domain[ql], cr - cl)]
            pmf: Self = self.from_pairs(pairs)
            assert pmf.total == self.total
            groups.append(pmf)

        # Cache and return result.
        quantiles = tuple(groups)
        self.__quantiles[nq] = quantiles
        return quantiles

    # ==================================================================
    # TABULATON AND OUTPUT

    def format_pairs(
        self,
        spec: str = "",
        /,
        *,
        scale: _Real = 1,
    ) -> Iterator[tuple[str, str]]:
        """Format PMF as a table."""
        # Validate & initialize parameters.
        if not len(self):
            return iter(tuple())
        specs = spec.split(":", 1)
        dspec: str = specs[0]
        ispec: str = specs[-1]  # Same as vspec if there's no colon.

        image: Iterator[_Real]
        if not ispec:
            image = (scale * p for p in self.image)
        elif ispec[-1] in "Xbcdnox":
            image = (scale * wt for wt in self.weights)
        else:
            image = (float(scale * p) for p in self.image)

        fdomain = (format(ev, dspec) for ev in self.domain)
        fimage = (format(p, ispec) for p in image)
        return zip(fdomain, fimage, strict=True)

    def plot(
        self,
        *,
        q: None | int | Auto = Ellipsis,
        xformat: str = "",
        yformat: str = ".2f",
        scale: _Real = 100,
        window_title: str = "bones",
        block: bool = True,
        console: bool = False,
        plotlib: str = "matplotlib",
    ) -> None:
        """Display the PMF with matplotlib."""
        # Common console/plotlib variables.
        fspec = ":".join((xformat, yformat))

        # Fall back to console if plotlib is unavailable or overridden.
        if importlib.util.find_spec(plotlib) is None:
            console = True
        if console:
            print("\n".join(self.tabulate(fspec, scale=scale)))
            return

        # Dynamically import plotlib for modularity & testability.
        plt = importlib.import_module(".pyplot", package=plotlib)
        patches = importlib.import_module(".patches", package=plotlib)

        # Set up plot.
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(window_title)

        domain = self.domain
        image = tuple(float(self(v)) for v in domain)
        n = len(domain)

        xlabels, ylabels = zip(*self.format_pairs(fspec, scale=scale), strict=True)
        legend: list[Any] = []

        # Group events into quantiles.
        nq = 0 if q is None else q if isinstance(q, int) else self.auto_quantile
        color1: list[ColorTriplet] = []  # stripe 1 color
        color2: list[ColorTriplet] = []  # stripe 2 color
        hatch: list[str] = []
        hatch_style = "/"
        hatch_width = 6.0 * math.sqrt(2)
        if 2 <= nq:
            # Set up quantile colors.
            qcolor = color_array(nq)
            for i in range(nq):
                # Label quantiles from 1/N to N/N.
                width = len(str(nq))
                fill = "\u2007"  # U+2007 FIGURE SPACE, &numsp;
                label = f"{1+i:{fill}>{width}d}/{nq}"
                legend.append(patches.Patch(color=qcolor[i], label=label))
            quantiles = self.quantile_groups(q or 0)
            for i in range(n):
                # Zero-weight events are black.
                if not self.weights[i]:
                    hatch.append("")
                    color1.append((0, 0, 0))
                    color2.append((0, 0, 0))
                    continue
                # Color events from blue to red to green.
                groups = [j for j in range(nq) if domain[i] in quantiles[j]]
                hatch.append("" if len(groups) == 1 else hatch_style)
                color1.append(qcolor[groups[0]])
                color2.append(qcolor[groups[-1]])
        else:
            # Color probabilities from violet to red.
            color1 = [
                interpolate_color(i, tmin=min(image), tmax=max(image)) for i in image
            ]
        edgecolor = [adjust_lightness(0.65, c) for c in color1]

        # Main bar colors.
        chart = ax.bar(
            x=range(n),
            height=image,
            tick_label=xlabels,
            color=color1,
            edgecolor=color2 or edgecolor,
            hatch=hatch or "",
        )
        # Edges only, to cover the hatch color.
        if color2:
            ax.bar(
                x=range(n),
                height=image,
                color="none",
                edgecolor=edgecolor,
            )
        # Labels and legend.
        ax.set_xlabel("Events")  # TODO: customizable title
        ax.set_ylabel("Probability")
        ax.bar_label(chart, labels=ylabels, padding=1, fontsize="x-small")
        if legend:
            ax.legend(
                title=quantile_name(nq),
                title_fontsize="small",
                handles=legend,
                fontsize="x-small",
            )

        # Show plot.
        with plt.rc_context({"hatch.linewidth": hatch_width}):
            plt.show(block=block)

    def tabulate(
        self,
        spec: str = "",
        /,
        *,
        scale: _Real = 1,  # TODO: switch to pair_format
        align: bool = True,
        separator: str | None = None,
    ) -> Iterator[str]:
        """Format PMF as a table."""
        # Validate & initialize parameters.
        if not self:
            return iter(tuple())
        if separator is None:
            separator = "  " if align else ": "

        xcol, ycol = zip(*self.format_pairs(spec, scale=scale), strict=True)
        xwidth = max(len(s) for s in xcol) if align else 0
        ywidth = max(len(s) for s in ycol) if align else 0
        # Align numbers right, everything else left.
        xjust = (
            x.rjust(xwidth)
            if isinstance(self.domain[i], numbers.Number)
            else x.ljust(xwidth)
            for i, x in enumerate(xcol)
        )
        yjust = (y.rjust(ywidth) for y in ycol)
        return (separator.join(pair) for pair in zip(xjust, yjust, strict=True))

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
        return NotImplemented

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
    # COLLECTION METHODS

    def __contains__(self, event: Any) -> bool:
        """Test object for membership in the event domain."""
        return event in self.mapping

    def __iter__(self) -> Iterator[ET_co]:
        """Iterate over the event domain."""
        return iter(self.domain)

    def __reversed__(self) -> Iterator[ET_co]:
        """Iterate over the event domain in reverse."""
        return reversed(self.domain)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.mapping)

    def index(self, event: Any, /) -> int:
        """Return the domain index of a given event."""
        try:
            return self.__index[event]
        except KeyError:
            raise ValueError(f"{event!r} not in {type(self).__name__}") from None

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
        0: (_t("{}-quantile"), _t("{}-quantiles")),
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
        0: (_t("{}-quantile"), _t("{}-quantiles")),
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
        0: (_t("{}-quantile"), _t("{}-quantiles")),
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
def quantile_name(size: int, /, *, kind: QK = QK.GROUP, plural: bool = True) -> str:
    """Return the name for a quantile of given size."""
    default = quantile_names[kind][0]
    name = quantile_names[kind].get(size, default)
    return name[int(plural)].format(size)
