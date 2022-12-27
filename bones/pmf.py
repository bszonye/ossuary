"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "BasePMF",
    "D",
    "D00",
    "D000",
    "D10",
    "D100",
    "D1000",
    "D12",
    "D2",
    "D20",
    "D3",
    "D30",
    "D4",
    "D6",
    "D8",
    "DF",
    "DiceTuplePMF",
    "PMF",
    "R00",
    "R000",
    "RF",
    "comb",
    "die_range",
    "multiset_comb",
    "multiset_perm",
    "perm",
]

import functools
import itertools
import math
import numbers
from collections import Counter
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
from typing import Any, cast, Optional, Self, SupportsIndex, TypeAlias, TypeVar

# PMF input types.
# TODO: Rename these to EventT and WeightT?
# TODO: Export the type vars in __all__?
# TODO: Remove Fraction (requires normalization rework).
ET_co = TypeVar("ET_co", bound=Hashable)  # Event type.
WT: TypeAlias = int  # Weight type.


class BasePMF(Collection[ET_co]):
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
        __events: Mapping[Any, WT] | Iterable[Any] = (),
        /,
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize PMF object."""
        # Convert input mapping or iterable to a list of pairs.
        pairs: Iterable[tuple[Any, WT]]
        match __events:
            case BasePMF() if type(__events) is type(self):
                # Copy another PMF of the same type.
                copy = cast(BasePMF[ET_co], __events)
                if normalize:
                    copy = copy.normalized()
                self.__weights = copy.__weights
                self.__total = copy.__total
                return
            case Mapping():
                pairs = cast(ItemsView[Any, Any], __events.items())
            case Iterable():
                pairs = ((event, 1) for event in __events)
            case _:
                raise TypeError(f"not iterable: {type(__events).__name__!r}")
        # Collect event weights.
        weights: dict[ET_co, WT] = {}
        total = 0
        for iv, iw in pairs:
            # Convert input values to events.  Subtypes override this.
            event: ET_co = self.init_event(iv)
            # Check parameter types and values.
            if not isinstance(event, Hashable):
                raise TypeError(f"unhashable type: {type(event).__name__!r}")
            if not isinstance(iw, int):
                raise TypeError(f"not a probability weight: {type(iw).__name__!r}")
            if iw < 0:
                raise ValueError(f"not a probability weight: {iw!r} < 0")
            # Record event weight and total weight.
            weights[event] = weights.setdefault(event, 0) + iw
            total += iw
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
    def init_event(cls, __value: Hashable, /) -> ET_co:
        """Check input values and convert them as needed."""
        return cast(ET_co, __value)  # Override this!

    @classmethod
    @functools.cache
    def die(cls, __faces: int | Sequence[ET_co] = 6, /) -> Self:
        """Generate the PMF for rolling one fair die with K faces."""
        faces: Self
        match __faces:
            case BasePMF():
                faces = cls(__faces)
            case int():
                faces = cls({item: 1 for item in die_range(__faces)})
            case Sequence():
                faces = cls(Counter(__faces))
            case _:
                raise TypeError(f"not a die: {type(__faces).__name__!r}")
        return faces

    @classmethod
    @functools.cache
    def enumerate_NdX(
        cls, dice: int, faces: int | Sequence[ET_co] = 6
    ) -> Iterable[tuple[ET_co, ...]]:
        """Generate all distinct dice pool combinations."""
        if dice < 1:
            return ()
        fpmf = cls.die(faces)
        # TODO: Should this also calculate pool weights, to handle the
        # case where faces is a PMF instead of a sequence?
        # TODO: Enumerate Counter objects instead of sequences?
        return tuple(itertools.combinations_with_replacement(fpmf, dice))

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
    def weight_graph(self) -> Sequence[tuple[ET_co, WT]]:
        """Return all of the (event, weight) pairs for the function."""
        return tuple(self.mapping.items())

    @property
    def mapping(self) -> Mapping[ET_co, WT]:
        """Provide read-only access to the probability mapping."""
        return self.__weights

    @property
    def total(self) -> WT:
        """Provide read-only access to the total probability."""
        return self.__total

    def mass(self, __event: ET_co, /) -> Fraction:
        """Return the probability mass of a given event."""
        return Fraction(self.mapping[__event], self.total)

    def weight(self, __event: ET_co, /) -> WT:
        """Return the probability weight of a given event."""
        return self.mapping[__event]

    def normalized(self) -> Self:
        """Return an equivalent object with minimum integer weights."""
        return type(self)(self, normalize=True)

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

        # Select probability weight or mass depending on format.  Use
        # integer weights for int formats, fractional masses otherwise.
        pfunc = self.weight if wspec and wspec[-1] in "Xbcdnox" else self.mass
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

    def __contains__(self, __event: object) -> bool:
        """Test object for membership in the event domain."""
        return bool(self.mapping.get(__event, 0))  # type: ignore

    def __call__(self, __event: ET_co) -> Fraction:
        """Return the given event probability as a fraction."""
        return Fraction(self.mapping.get(__event, 0), self.total)

    def __iter__(self) -> Iterator[ET_co]:
        """Iterate over the event domain."""
        return iter(self.domain)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.mapping)


DieFaces = int | Sequence[Any]  # TODO: Any -> Hashable?


class PMF(BasePMF[Hashable]):
    """PMF for any Hashable value."""


def die_range(__arg1: int, __arg2: Optional[int] = None, /) -> range:
    """Create a range over the numbered faces of a die.

    This is a convenience function to make it easier to declare ranges
    matching the faces on dice.  Unlike the standard range function, the
    stop value is inclusive, the default start value is 1, and there is
    no option to skip values.  The function infers the step direction
    from the stop and start values.
    """
    match __arg1, __arg2:
        case int() as stop, None:
            return range(1, stop + 1) if 0 <= stop else range(-stop, 0, -1)
        case int() as start, int() as stop:
            step = 1 if start <= stop else -1
            return range(start, stop + step, step)
        case SupportsIndex() as start, int() as stop:
            # Cannot infer direction if start is not an int.
            return range(start, stop + 1)
        case SupportsIndex(), fail:
            pass
        case fail, _:
            pass
    raise TypeError(
        f"{type(fail).__name__!r} object cannot be interpreted as an integer"
    )


# TODO: allow other Hashable dice values?
DiceIterable = Iterable[int]
DiceSequence = Sequence[int]
DiceTuple = tuple[int, ...]


@functools.cache
def comb(__n: int, __k: int, /) -> int:
    """Cache results from the math.comb function."""
    return math.comb(__n, __k)


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
    k = sorted(tuple(__iter), reverse=True)
    n = sum(k)
    if not n:
        return 0
    # Use N! / k[0]! as a starting point to take advantage of
    # optimizations in the math.perm function.
    weight = perm(n, n - k[0])
    # Divide the running product by k[n]! for each other subgroup.
    for count in k[1:]:
        weight //= math.factorial(count)
    return weight


@functools.cache
def perm(__n: int, __k: Optional[int] = None, /) -> int:
    """Cache results from the math.perm function."""
    return math.perm(__n, __k)


@functools.cache
def pmf_weight(__dice: DiceIterable) -> int:
    """Determine the PMF weight of a DiceTuple.

    Given a dice tuple, determine how many ways there are to arrange
    the dice into equivalent rolls.  For example, there are six ways
    to arrange (3, 2, 1), but there are only three arrangements of
    (6, 6, 1).  This number is useful for generating PMF weights for
    groups of equivalent dice rolls.  For example:

    {(roll, roll.weight) for roll in enumerate_NdX(dice=dice)}
    """
    # TODO: Accept PMFs to handle non-uniform value weights?
    # TODO: Remove this?
    counts = tuple(sorted(Counter(__dice).values(), reverse=True))
    return multiset_perm(counts)


class DiceTuplePMF(BasePMF[DiceTuple]):
    """Probability mass function for analyzing raw dice pools.

    Use this when you need to analyze or select specific dice rolls
    within a dice pool.  For example, use it to analyze mechanics like
    "roll 4d6, dropping the lowest die."  You don't need this class to
    count successes in a dice pool.
    """

    @classmethod
    def init_event(cls, __value: Hashable, /) -> DiceTuple:
        """Check input values and convert them as needed."""
        failtype = ""
        match __value:
            case int():
                __value = (__value,)
            case Iterable() as items:
                # TODO: pyright wants this cast. can we do better?
                __value = tuple(cast(Iterable[Hashable], items))
                for die in __value:
                    # TODO: allow other Hashable die values?
                    if not isinstance(die, int):
                        failtype = type(die).__name__
                        break
            case _:
                failtype = type(__value).__name__
        if failtype:
            raise TypeError(f"not a dice tuple: {failtype!r}")
        return super().init_event(__value)

    @classmethod
    @functools.cache
    def NdX(cls, dice: int = 1, faces: DieFaces = 6) -> Self:
        """Create the PMF for rolling a pool of N dice with K faces."""
        enumeration = PMF.enumerate_NdX(dice, faces)
        return cls((pool, pmf_weight(pool)) for pool in enumeration)

    @classmethod
    @functools.cache
    def NdX_select(
        cls,
        dice: int = 1,
        faces: DieFaces = 6,
        *,
        dh: int = 0,
        dl: int = 0,
        kh: int = 0,
        kl: int = 0,
        km: int = 0,
    ) -> Self:
        """Create a PMF for NdX, keeping the highest N of M dice."""
        # Validate & initialize parameters.
        if min(dh, dl, kh, kl, km) < 0:
            raise ValueError("negative die selector")
        keep = max(kh, kl, km)
        if keep != kh + kl + km:
            raise ValueError("too many keep selectors")

        # Determine the number of dice left after the drop selectors,
        # then the final number kept after the keep selectors.
        leave = max(dice - dh - dl, 0)
        keep = min(keep, leave) if keep else leave
        if keep <= 0:
            return cls()

        # Convert the keep selector to the equivalent drop selectors by
        # adding any extra dice to dl or dh as appropriate.  In the case
        # of an uneven median, assign the extra drop to the high dice.
        dx = leave - keep
        if km:
            dl += dx // 2
            dh += (dx + 1) // 2
        elif kh:
            dl += dx
        elif kl:
            dh += dx

        # Enumerate combinations for the remaining dice and determine
        # how many ways you can permute each one.
        fpmf = PMF.die(faces)
        weights = list(fpmf.weights) + [0, 0]
        faces = tuple(fpmf)
        nfaces = len(faces)
        # Symbolic constants outside of range(nfaces), used below.
        H = nfaces  # Represents any die higher than the selected dice.
        L = H + 1  # Likewise, but for lower dice.
        # TODO: Clean up debug code.
        # print(f"{dx=} {dh=} {dl=} {weights=}")

        # Enumerate the faces by ordinality rather than face value, to
        # preserve the input order and to accommodate non-comparable die
        # values.  For example, this enumerates three six-sided dice
        # numerically from (0, 0, 0) to (5, 5, 5).
        pweight: dict[Sequence[Hashable], WT] = {}
        for ipool in itertools.combinations_with_replacement(range(nfaces), keep):
            # Get the range of face numbers.
            low = ipool[0]
            high = ipool[-1]
            # How many faces are before low or after high?
            nlow = low
            nhigh = nfaces - 1 - high
            # What are the total weights of the low and high dice?
            weights[L] = sum(weights[i] for i in range(nlow))
            weights[H] = sum(weights[nfaces - 1 - i] for i in range(nhigh))
            # print(f"{low=} {nlow=} {high=} {nhigh=} {weights=}")

            # Calculate all of the combinations of _unselected_ dice.
            # Example: If we are calculating 6d6kh3, and the selected
            # dice are (1, 2, 3), there's only one possible combination
            # of the lower dice:
            #
            # * (1, 1, 1, 1, 2, 3)
            #
            # However, if the selected dice are (4, 5, 6), then there
            # are many possible combinations of the lower dice:
            #
            # * (L, L, L, 4, 5, 6)
            # * (L, L, 4, 4, 5, 6)
            # * (L, 4, 4, 4, 5, 6)
            # * (4, 4, 4, 4, 5, 6)
            #
            # Each L represents (nlow == 3) possible faces that we don't
            # need to enumerate explicitly.  Instead, we can multiply
            # the weight by nlow for each L in the combination.
            #
            # We can count the highest dice in a keep-low/drop-high dice
            # in the same way, by replacing all of the highest values
            # with the symbol H and counting each one nhigh times.
            base_counter = Counter(ipool)
            weight = 0
            low_range = range(1 + (dl if nlow else 0))
            high_range = range(1 + (dh if nhigh else 0))
            for i in low_range:
                for j in high_range:
                    counter = base_counter.copy()
                    counter[L] = i
                    counter[H] = j
                    counter[low] += dl - i
                    counter[high] += dh - j
                    # Count multiset permutations.
                    counts = tuple(sorted(counter.values(), reverse=True))
                    cperm = multiset_perm(counts)
                    # cweight = math.prod(weights[k] ** n for k, n in counter.items())
                    cweight = math.prod(weights[k] for k in counter.elements())
                    # print(f"{ipool=} {i=} {j=} {cperm=} {cweight=}")
                    # print(tuple(counter.elements()))
                    weight += cweight * cperm

            # Translate ordinals to face values and base weights.
            vpool = tuple(faces[p] for p in ipool)
            pweight[vpool] = weight
            # print(f"{ipool=} {vpool=} {weight=} {low=} {high=} {dl=} {dh=}")
        return cls(pweight)

    def sum_pools(self) -> PMF:
        """Sum the pools and return the resulting PMF."""
        pmf: dict[int, WT] = {}
        for pool, count in self.mapping.items():
            total = sum(tuple(pool))
            pmf.setdefault(total, 0)
            pmf[total] += count
        return PMF(pmf)


# Call D(K) to create the PMF for rolling 1dX.
D = PMF.die

# Common die sizes.
D2 = D(2)
D3 = D(3)
D4 = D(4)
D6 = D(6)
D8 = D(8)
D10 = D(10)
D12 = D(12)
D20 = D(20)
D30 = D(30)
D100 = D(100)
D1000 = D(1000)

# Special die sizes.
R00 = die_range(0, 99)
D00 = D(R00)  # D100 counted from zero.
R000 = die_range(0, 999)
D000 = D(R000)  # D1000 counted from zero.
RF = die_range(-1, +1)
DF = D(RF)  # Fate/Fudge dice.
