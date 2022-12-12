"""bones.pmf: probability mass function type."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from fractions import Fraction
from types import MappingProxyType
from typing import cast, Optional, Self, TypeVar, Union

Probability = Union[int, Fraction]

# PMF input types.
_DVT = TypeVar("_DVT", bound=Hashable)
PairT = tuple[_DVT, Optional[Probability]]
IterableT = Iterable[Union[_DVT, PairT[_DVT]]]
MappingT = Mapping[_DVT, Optional[Probability]]


class PMF(Mapping[Hashable, Probability]):
    """Finite probability mass function."""

    ValueT = Hashable
    ValueType: type = Hashable
    ValueInit: Optional[Callable[..., Hashable]] = None

    __pairs: Mapping[Hashable, Probability]
    __total: Probability  # TODO: Is this redundant?

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
                # TODO: Handle denominator and normalize.
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
            if not isinstance(v, self.ValueType):
                # Try to convert the value.
                converter = self.ValueInit
                if converter is None:
                    vtype = self.ValueType.__name__
                    vactual = type(v).__name__
                    raise TypeError(f"{vactual!r} object is not {vtype!r}")
                else:
                    v = converter(v)
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

    @property
    def pairs(self) -> Mapping[Hashable, Probability]:
        """Provide read-only access to the probability mapping."""
        return self.__pairs

    @property
    def total(self) -> Probability:
        """Provide read-only access to the total probability."""
        return self.__total

    def __getitem__(self, key: Hashable) -> Probability:
        """Return the probability for a given value."""
        return self.__pairs[key]

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the discrete values."""
        return iter(self.__pairs)

    def __len__(self) -> int:
        """Return the number of discrete values in the mapping."""
        return len(self.__pairs)
