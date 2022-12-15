"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import functools
import itertools
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType
from typing import cast, Optional, Self, TypeVar, Union

# TODO: Allow Decimal or Rational too?
DiceValue = Union[int, Fraction]
Probability = Union[int, Fraction]

# PMF input types.
_DVT = TypeVar("_DVT", bound=Hashable)
PairT = tuple[_DVT, Optional[Probability]]
IterableT = Union[Iterable[_DVT], Iterable[PairT[_DVT]]]
MappingT = Mapping[_DVT, Optional[Probability]]
DieRange = Union[int, range]


class PMF(Mapping[Hashable, Probability]):
    """Finite probability mass function."""

    __pairs: Mapping[Hashable, Probability]
    __total: Probability

    def __init__(
        self,
        __items: Union[Self, MappingT[Hashable], IterableT[Hashable]] = (),
        /,
        *,
        denominator: Probability = 0,
        normalize: bool = False,
    ) -> None:
        """Initialize PMF object."""
        items: list[tuple[Hashable, Optional[Probability]]] = []
        match __items:
            case PMF():
                if denominator or normalize:
                    items = list(__items.items())
                else:
                    self.__pairs = __items.__pairs
                    self.__total = __items.__total
                    return
            case Mapping():
                mapping = cast(MappingT[Hashable], __items)
                for mv, mp in mapping.items():
                    match mp:
                        case int() | Fraction() | None:
                            items.append((mv, mp))
                        case _:  # Verify mapping cast.
                            raise TypeError(f"not a probability: {mp!r}")
            case Iterable():
                for item in __items:
                    match item:
                        case [Hashable() as iv, int() | Fraction() | None as ip]:
                            items.append((iv, ip))
                        case [iv, int() | Fraction() | None as p]:
                            raise TypeError(f"unhashable type: {type(iv).__name__!r}")
                        case Hashable() as iv:
                            items.append((iv, 1))
                        case _:
                            raise TypeError(
                                f"unhashable type: {type(item).__name__!r}"
                            )
            case _:
                raise TypeError(f"{type(__items).__name__!r}")
        # Collect pairs.
        pairs: dict[Hashable, Probability] = {}
        total: Probability = 0
        remainder: Probability = 0
        remainder_set: set[Hashable] = set()
        for item in items:
            match item:
                case [v, None]:
                    # Add v to the remainder set.
                    remainder_set.add(v)
                    p = 0
                case [v, int() | Fraction() as p]:
                    pass
                case _:
                    v = item
                    p = 1
            v = self.validate_value(v)  # Subtypes override this.
            pairs.setdefault(v, 0)
            pairs[v] += p
            total += p
        # Determine & distribute remainder, if any.
        if denominator:
            remainder = denominator - total
        if remainder:
            if not remainder_set:
                raise ValueError(
                    f"total probability {total} < denominator {denominator}"
                )
            total = denominator
            shares = len(remainder_set)
            if shares != 1:
                remainder = Fraction(remainder, shares)
            for v in remainder_set:
                pairs[v] += remainder
        # Optionally normalize total probability to 1.
        if normalize and total != 1:
            pairs = {v: Fraction(p, total) for v, p in pairs.items()}
            total = 1
        # Initialize attributes.
        self.__pairs = MappingProxyType(pairs)
        self.__total = total

    @classmethod
    def validate_value(cls, __value: Hashable, /) -> Hashable:
        """Check input values and convert them as needed."""
        return __value

    @property
    def denominator(self) -> Probability:
        """Provide read-only access to the total probability."""
        return self.__total

    @property
    def pairs(self) -> Mapping[Hashable, Probability]:
        """Provide read-only access to the probability mapping."""
        return self.__pairs

    @property
    def support(self) -> Sequence[Hashable]:
        """Return all of the values with nonzero probability."""
        return tuple(v for v, p in self.__pairs.items() if p)

    def normalized(self) -> Self:
        """Normalize denominator to 1 and return the result."""
        return self if self.__total == 1 else type(self)(self, normalize=True)

    def __getitem__(self, key: Hashable) -> Probability:
        """Return the probability for a given value."""
        return self.__pairs[key]

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the discrete values."""
        return iter(self.__pairs)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.__pairs)

    def __format__(self, spec: str) -> str:
        """Format the PMF according to the format spec."""
        out = "{\n"
        for value, weight in self.__pairs.items():
            fw: str
            if spec:
                fw = format(float(Fraction(weight, self.__total)), spec)
            else:
                fw = str(weight)
            out += f"    {value}: {fw},\n"
        out += "}"
        return out

    def __str__(self) -> str:
        """Format the PMF for printing."""
        return self.__format__("")


class DicePMF(PMF):
    """Probability mass function for dice rolls."""

    def __init__(
        self,
        __items: Union[Self, MappingT[DiceValue], IterableT[DiceValue]] = (),
        /,
        denominator: Probability = 0,
        normalize: bool = False,
    ) -> None:
        """Initialize object via super."""
        super().__init__(
            __items,
            denominator=denominator,
            normalize=normalize,
        )

    @classmethod
    def validate_value(cls, __value: Hashable, /) -> Hashable:
        """Check input values and convert them as needed."""
        match __value:
            case int() | Fraction():
                pass
            case float() | str():
                __value = Fraction(__value)
            case [int() as numerator, int() as denominator]:
                __value = Fraction(numerator, denominator)
            case _:
                raise TypeError(f"irrational type: {type(__value).__name__!r}")
        return super().validate_value(__value)

    @classmethod
    @functools.cache
    def roll_1dK(cls, __sides: DieRange, /) -> Self:
        """Generate the PMF for rolling one fair die with K sides."""
        faces = die_range(__sides) if isinstance(__sides, int) else __sides
        return cls(faces)

    # Override type signatures for methods returning Hashable.
    @property
    def pairs(self) -> Mapping[DiceValue, Probability]:  # type: ignore
        """Provide read-only access to the probability mapping."""
        return cast(Mapping[DiceValue, Probability], super().pairs)

    @property
    def support(self) -> Sequence[DiceValue]:
        """Return all of the values with nonzero probability."""
        return cast(Sequence[DiceValue], super().support)

    def __iter__(self) -> Iterator[DiceValue]:
        """Iterate over the discrete values."""
        return cast(Iterator[DiceValue], super().__iter__())


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


class DiceTuplePMF(PMF):
    """Probability mass function for analyzing raw dice pools.

    Use this when you need to analyze or select specific dice rolls
    within a dice pool.  For example, use it to analyze mechanics like
    "roll 4d6, dropping the lowest die."  You don't need this class to
    count successes in a dice pool.
    """

    def __init__(
        self,
        __items: Union[Self, MappingT[DiceTuple], IterableT[DiceTuple]] = (),
        /,
        denominator: Probability = 0,
        normalize: bool = False,
    ) -> None:
        """Initialize object via super."""
        super().__init__(
            __items,
            denominator=denominator,
            normalize=normalize,
        )

    @classmethod
    def validate_value(cls, __value: Hashable, /) -> Hashable:
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
            raise TypeError(f"not convertible to dice tuple: {failtype!r}")
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

    # Override type signatures for methods returning Hashable.
    @property
    def pairs(self) -> Mapping[DiceTuple, Probability]:  # type: ignore
        """Provide read-only access to the probability mapping."""
        return cast(Mapping[DiceTuple, Probability], super().pairs)

    @property
    def support(self) -> Sequence[DiceTuple]:
        """Return all of the values with nonzero probability."""
        return cast(Sequence[DiceTuple], super().support)

    def __iter__(self) -> Iterator[DiceTuple]:
        """Iterate over the discrete values."""
        return cast(Iterator[DiceTuple], super().__iter__())


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
_R00 = die_range(0, 99)
D00 = D(_R00)  # D100 counted from zero.
_R000 = die_range(0, 999)
D000 = D(_R000)  # D1000 counted from zero.
_RF = die_range(-1, +1)
DF = D(_RF)  # Fate/Fudge dice.
