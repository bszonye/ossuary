"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import functools
import itertools
import math
import numbers
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast, Optional, Self, TypeVar, Union

DieRange = Union[int, range]
DiceValue = Union[int, Fraction]
Probability = Union[int, Fraction]

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
        __items: Union[MappingSpec[Any], Iterable[PairSpec[Any]]] = (),
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
            # Fractions don't support format specs, and it's not obvious
            # how they'd work.  For now just convert to float.
            if spec and isinstance(item, Fraction):
                item = float(item)

            # Simple alignment: numbers right, everything else left.
            # Use explicit alignment + sufficient width to override.
            text = format(item, spec)
            if isinstance(item, numbers.Number):
                text = text.rjust(width)
            else:
                text = text.ljust(width)
            return text

        def measure(items: Iterable[Any], spec: str) -> int:
            if not align:  # Use minimum width.
                return 0
            width = 0  # Greatest overall width.
            for item in items:
                text = column(item, spec)
                width = max(width, len(text))
            return width

        # Normalize weights.
        pmf = self.normalized(0 if wspec and wspec[-1] in "Xbcdnox" else 1)
        # Determine the minimum column & fraction widths.
        vwidth = measure(pmf.keys(), vspec)
        wwidth = measure(pmf.values(), wspec)
        # Generate text.
        return tuple(
            separator.join(
                (
                    column(value, vspec, vwidth),
                    column(weight, wspec, wwidth),
                )
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

    @classmethod
    @functools.cache
    def roll_1dK(cls, __sides: DieRange, /) -> Self:
        """Generate the PMF for rolling one fair die with K sides."""
        faces = die_range(__sides) if isinstance(__sides, int) else __sides
        pairs = ((face, 1) for face in faces)
        return cls(pairs)


def die_range(__arg1: int, __arg2: Optional[int] = None, /) -> range:
    """Create a range over the numbered faces of a die.

    This is a convenience function to make it easier to declare ranges
    matching the faces on dice.  Unlike the standard range function, the
    stop value is inclusive, the default start value is 1, and there is
    no option to skip values.
    """
    return range(1, __arg1 + 1) if __arg2 is None else range(__arg1, __arg2 + 1)


# TODO: allow other Hashable dice values?
DiceIterable = Iterable[int]
DiceSequence = Sequence[int]
DiceTuple = tuple[int, ...]


@functools.cache
def enumerate_NdK(dice: int, *, sides: DieRange = 6) -> Iterable[DiceTuple]:
    """Generate all distinct dice pool combinations."""
    if dice < 1:
        return ()
    faces = die_range(sides) if isinstance(sides, int) else sides
    return (
        tuple(pool) for pool in itertools.combinations_with_replacement(faces, dice)
    )


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
    k = sorted(__iter)
    n = sum(k)
    if not n:
        return 0
    # Use N! / k[0]! as a starting point to take advantage of
    # optimizations in the math.perm function.
    weight = math.perm(n, n - k[0])
    # Divide the running product by k[n]! for each other subgroup.
    for count in k[1:]:
        weight //= math.factorial(count)
    return weight


@functools.cache
def pmf_weight(__dice: DiceIterable) -> int:
    """Determine the PMF weight of a DiceTuple.

    Given a dice tuple, determine how many ways there are to arrange
    the dice into equivalent rolls.  For example, there are six ways
    to arrange (3, 2, 1), but there are only three arrangements of
    (6, 6, 1).  This number is useful for generating PMF weights for
    groups of equivalent dice rolls.  For example:

    {(roll, roll.weight) for roll in enumerate_NdK(dice=dice)}
    """
    counts = tuple(sorted(Counter(__dice).values()))
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
    def roll_NdK(cls, dice: int = 1, *, sides: DieRange = 6) -> Self:
        """Create the PMF for rolling a pool of N dice with K sides."""
        return cls((p, pmf_weight(p)) for p in enumerate_NdK(dice=dice, sides=sides))

    @classmethod
    @functools.cache
    def roll_NdK_keep(
        cls, dice: int = 1, *, sides: DieRange = 6, keep: int = 1
    ) -> Self:
        """Create a PMF for NdK, keeping the highest or lowest dice."""
        # Normalize parameters.
        nkeep = min(abs(keep), dice)  # Can't keep more than we roll.
        if nkeep < 1:
            return cls()
        # Start with the PMF for NdK where N = the number of kept dice.
        faces = die_range(sides) if isinstance(sides, int) else sides
        pmf = cls.roll_NdK(nkeep, sides=faces)
        # Merge additional dice with the given filtering rule.
        for _ in range(dice - nkeep):
            pmf = pmf.merge_die(faces, keep)
        return cls(pmf)

    def merge_die(self, sides: DieRange = 6, /, keep: int = 0) -> Self:
        """Add a die to the tuple, keeping if it passes the filter."""
        faces = die_range(sides) if isinstance(sides, int) else sides
        pmf: dict[DiceTuple, Probability] = {}
        for pool, pweight in self.pairs.items():
            for face in faces:
                # Add the new die to the tuple, then remove the highest
                # or lowest die if keep is nonzero.
                mlist = sorted(pool + (face,))
                if 0 < keep:  # Drop lowest.
                    mlist = mlist[1:]
                elif keep < 0:  # Drop highest.
                    mlist = mlist[:-1]
                mpool = DiceTuple(mlist)
                # Record the weight for the new tuple.
                pmf.setdefault(mpool, 0)
                pmf[mpool] += pweight
        return type(self)(pmf)

    def sum_pools(self) -> DicePMF:
        """Sum the pools and return the resulting DicePMF."""
        pmf: dict[int, Probability] = {}
        for pool, count in self.pairs.items():
            total = sum(tuple(pool))
            pmf.setdefault(total, 0)
            pmf[total] += count
        return DicePMF(pmf)


# Call D(K) to create the PMF for rolling 1dK.
D = DicePMF.roll_1dK

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
