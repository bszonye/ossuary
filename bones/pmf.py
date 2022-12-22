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
    "DicePMF",
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
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast, Optional, Self, SupportsIndex, TypeVar

DiceValue = int | Fraction
Probability = int | Fraction

# PMF input types.
_DVT = TypeVar("_DVT", bound=Hashable)
PairSpec = tuple[_DVT, Probability]
MappingSpec = Mapping[_DVT, Probability]


class BasePMF(Mapping[_DVT, Probability]):
    """Generic base class for finite probability mass functions."""

    __pweight: Mapping[_DVT, Probability]
    __ptotal: Probability

    def __init__(
        self,
        __items: MappingSpec[Any] | Iterable[PairSpec[Any]] = (),
        /,
        *,
        normalize: Probability = 0,
    ) -> None:
        """Initialize PMF object."""
        if normalize < 0:  # Reserve negative sizes for future expansion.
            raise ValueError(f"{normalize=} < 0")
        # Convert input mapping or iterable to a list of pairs.
        items: list[PairSpec[Any]]
        match __items:
            case BasePMF() if not normalize and type(__items) is type(self):
                # Copy another PMF of the same type.
                copy = cast(BasePMF[_DVT], __items)
                self.__pweight = copy.__pweight
                self.__ptotal = copy.__ptotal
                return
            case Mapping():
                # Get values and weights from a mapping.
                mapping = cast(MappingSpec[Any], __items)
                items = list(mapping.items())
            case Iterable():
                # Get values and weights by iterating over pairs.
                items = list(__items)
            case _:
                raise TypeError(f"not iterable: {type(__items).__name__!r}")
        # Collect value weights.
        pweight: dict[_DVT, Probability] = {}  # Probability weights.
        rweight: dict[_DVT, Probability] = {}  # Remainder weights.
        for item in items:
            match item:  # Check input structure.
                case [xvalue, int() | Fraction() as weight]:
                    pass
                case [_, xweight]:
                    raise TypeError(f"not a probability: {type(xweight).__name__!r}")
                case _:
                    raise TypeError(f"not a pair: {type(item).__name__!r}")
            value = self.validate_value(xvalue)  # Subtypes override this.
            if not isinstance(value, Hashable):
                raise TypeError(f"unhashable type: {type(value).__name__!r}")
            if weight < 0:
                rweight.setdefault(value, 0)
                rweight[value] -= weight
                weight = 0
            pweight.setdefault(value, 0)
            pweight[value] += weight
        # Determine the normalized weight and remainder.
        ptotal: Probability = sum(pweight.values())
        rtotal: Probability = sum(rweight.values())
        if not normalize:
            normalize = ptotal or 1  # Ensure a nonzero denominator.
        remainder: Probability = normalize - ptotal
        # Distribute the remainder, if there are any flex weights.
        if remainder and rtotal:
            scale = Fraction(remainder, rtotal)
            for v, w in rweight.items():
                rpw: Probability = scale * w
                pweight[v] += rpw
            ptotal = sum(pweight.values())
        # Scale weights to the normalized total.
        scale = Fraction(normalize, ptotal or 1)
        if scale != 1:
            for v, w in pweight.items():
                pweight[v] = scale * w
            ptotal = sum(pweight.values())
        # Reduce integral fractions.
        for v, w in pweight.items():
            numerator, denominator = w.as_integer_ratio()
            if denominator == 1:
                pweight[v] = numerator
        # Initialize attributes.
        self.__pweight = MappingProxyType(pweight)
        self.__ptotal = ptotal or normalize

    @classmethod
    def validate_value(cls, __value: Hashable, /) -> _DVT:
        """Check input values and convert them as needed."""
        return cast(_DVT, __value)  # Override this!

    @classmethod
    @functools.cache
    def die(cls, __faces: int | Sequence[_DVT] = 6, /) -> Self:
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
        cls, dice: int, faces: int | Sequence[_DVT] = 6
    ) -> Iterable[tuple[_DVT, ...]]:
        """Generate all distinct dice pool combinations."""
        if dice < 1:
            return ()
        fpmf = cls.die(faces)
        # TODO: Should this also calculate pool weights, to handle the
        # case where faces is a PMF instead of a sequence?
        # TODO: Enumerate Counter objects instead of sequences?
        return tuple(itertools.combinations_with_replacement(fpmf, dice))

    @property
    def int_weight(self) -> int:
        """Find the minimum total weight to avoid fractional weights."""
        # First, get all of the non-zero weights and sum them.
        weights = [w for w in self.values() if w]
        total = sum(weights)
        if not total:
            return 1

        # Next, determine whether we need to multiply out fractions.
        fracs = [w.denominator for w in weights]
        lcm = math.lcm(*fracs)

        # Resize based on the gcd (if integers) or lcm (if fractions).
        size: Probability
        if lcm == 1:
            nums = [w.numerator for w in weights]
            gcd = math.gcd(*nums)
            size = Fraction(total, gcd)
        else:
            size = Fraction(lcm * total, 1)

        # Return the integral size.
        assert size.denominator == 1
        return size.numerator

    @property
    def total_weight(self) -> Probability:
        """Provide read-only access to the total probability."""
        return self.__ptotal

    @property
    def pairs(self) -> MappingSpec[_DVT]:
        """Provide read-only access to the probability mapping."""
        return self.__pweight

    @property
    def support(self) -> Sequence[_DVT]:
        """Return all of the values with nonzero probability."""
        return tuple(v for v, p in self.__pweight.items() if p)

    def normalized(self, __total: Probability = 0, /) -> Self:
        """Normalize to a given total weight and return the result."""
        if __total < 0:  # Reserve negative sizes for future expansion.
            raise ValueError(f"total weight {__total} < 0")
        if not __total:
            __total = self.int_weight
        return (
            self if self.__ptotal == __total else type(self)(self, normalize=__total)
        )

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

        # Normalize weights.
        pmf = self.normalized(0 if wspec and wspec[-1] in "Xbcdnox" else 1)
        # Determine the minimum column & fraction widths.
        vwidth = measure(pmf.keys(), vspec)
        wwidth = measure(pmf.values(), wspec)
        # Generate text.
        return tuple(
            separator.join(
                (column(value, vspec, vwidth), column(weight, wspec, wwidth))
            )
            for value, weight in pmf.items()
        )

    def __format__(self, spec: str) -> str:
        """Format the PMF according to the format spec."""
        rows = self.tabulate(spec, align=False)
        return "{" + ", ".join(rows) + "}"

    def __repr__(self) -> str:
        """Format the PMF for diagnostics."""
        params = (
            repr(dict(self.__pweight))
            if self.__pweight
            else f"normalize={self.__ptotal!r}"
            if self.__ptotal != 1
            else ""
        )
        return f"{type(self).__name__}({params})"

    def __str__(self) -> str:
        """Format the PMF for printing."""
        return self.__format__("")

    def __getitem__(self, key: _DVT) -> Probability:
        """Return the probability for a given value."""
        return self.__pweight[key]

    def __iter__(self) -> Iterator[_DVT]:
        """Iterate over the discrete values."""
        return iter(self.__pweight)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.__pweight)


DieFaces = int | Sequence[Any]  # TODO: Any -> Hashable?


class PMF(BasePMF[Hashable]):
    """PMF for any Hashable value."""


class DicePMF(BasePMF[DiceValue]):
    """PMF for numeric values derived from dice rolls."""

    @classmethod
    def validate_value(cls, __value: Hashable, /) -> DiceValue:
        """Check input values and convert them as needed."""
        match __value:
            case int() | Fraction():
                pass
            case float() | str():
                __value = Fraction(__value)
            case [int() as numerator, int() as denominator]:
                # TODO: test this
                # TODO: convert integral fractions to int?
                __value = Fraction(numerator, denominator)
            case _:
                raise TypeError(f"irrational type: {type(__value).__name__!r}")
        return super().validate_value(__value)


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
    def validate_value(cls, __value: Hashable, /) -> DiceTuple:
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
        return super().validate_value(__value)

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
        weights = [cast(int, w) for w in fpmf.values()] + [0, 0]
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
        pweight: dict[Sequence[Hashable], Probability] = {}
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
        pmf: dict[int, Probability] = {}
        for pool, count in self.pairs.items():
            total = sum(tuple(pool))
            pmf.setdefault(total, 0)
            pmf[total] += count
        return PMF(pmf)


# Call D(K) to create the PMF for rolling 1dX.
D = DicePMF.die

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
