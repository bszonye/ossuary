"""bones.roll: dice roll analysis."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "Dice",
    "Die",
    "d",
    "d00",
    "d000",
    "d10",
    "d100",
    "d1000",
    "d12",
    "d2",
    "d20",
    "d3",
    "d30",
    "d4",
    "d6",
    "d8",
    "dF",
]

import functools
import itertools
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import cast, overload, Self, TypeVar

from .pmf import multiset_perm, PMF, Weight

ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.
_T = TypeVar("_T")


class Die(PMF[ET_co]):
    """Model for rolling a die."""

    # Note: The ET_co overload needs to come first for mypy to infer
    # types correctly for Die.from_pairs and similar classmethods.
    @overload
    def __init__(
        self, faces: Iterable[ET_co], /, *, reverse: bool = False
    ) -> None:  # noqa: D107
        ...

    @overload
    def __init__(
        self: "Die[int]", faces: int = ..., /, *, reverse: bool = False
    ) -> None:  # noqa: D107
        ...

    def __init__(
        self, faces: Iterable[ET_co] | int = 6, /, *, reverse: bool = False
    ) -> None:
        """Initialze the PMF for a die with the given faces."""
        match faces:
            case PMF():
                self.from_self(faces, instance=self, reverse=reverse, normalize=False)
            case int():
                # Copy self.d(faces) to share internals if possible.
                die = cast(Self, self.d(faces, reverse=reverse))
                self.from_self(die, instance=self, normalize=False)
            case _:
                self.from_iterable(
                    faces, instance=self, reverse=reverse, normalize=False
                )

    @classmethod
    @functools.cache
    def d(
        cls: "type[Die[int]]",
        n: int = 6,
        /,
        *,
        reverse: bool = False,
    ) -> "Die[int]":
        """Create a die with faces numbered from 1 to n."""
        sides = int(n)
        faces = range(1, 1 + sides)
        pairs: Iterator[tuple[int, Weight]] = ((event, 1) for event in faces)
        if reverse:
            pairs = reversed(tuple(pairs))
        return cls.from_pairs(pairs, normalize=False)

    def __repr__(self) -> str:
        """Format the PMF for diagnostics."""
        parameters = (
            # Show the full population if it's reasonable.
            tuple(self.population())
            if self.modal_weight < 4
            # Otherwise, fall back to map representation.
            else self.mapping
        )
        return f"{type(self).__name__}({parameters})"


# Call d(K) to create the PMF for rolling 1dX.
d = Die.d

# Common die sizes.
d2 = d(2)
d3 = d(3)
d4 = d(4)
d6 = d(6)
d8 = d(8)
d10 = d(10)
d12 = d(12)
d20 = d(20)
d30 = d(30)
d100 = d(100)
d1000 = d(1000)

d00 = d(100) - 1  # D100 counted from zero.
d000 = d(1000) - 1  # D1000 counted from zero.
dF = d(3) - 2  # Fate/Fudge dice.


class Dice(PMF[tuple[ET_co, ...]]):
    """Probability mass function for analyzing raw dice rolls.

    Use this when you need to analyze or select specific dice out of
    many dice rolled.  For example, use it to analyze mechanics like
    "roll 4d6, dropping the lowest die."  You don't need this class to
    count successes in a dice pool.
    """

    @classmethod
    def NdX(
        cls,
        n: int,
        die: Die[_T],
        /,
        *,
        dh: int = 0,
        dl: int = 0,
        kh: int = 0,
        kl: int = 0,
        km: int = 0,
    ) -> "Dice[_T]":
        """Create a PMF for NdX, keeping the highest N of M dice."""
        # Validate & initialize parameters.
        if min(dh, dl, kh, kl, km) < 0:
            raise ValueError("negative die selector")
        keep = max(kh, kl, km)
        if keep != kh + kl + km:
            raise ValueError("too many keep selectors")

        # Determine the number of dice left after the drop selectors,
        # then the final number kept after the keep selectors.
        leave = max(n - dh - dl, 0)
        keep = min(keep, leave) if keep else leave
        if keep <= 0:
            return cast(Dice[_T], cls).from_pairs(())

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
        weights = list(die.weights) + [0, 0]
        faces = tuple(die)
        nfaces = len(faces)
        # Symbolic constants outside of range(nfaces), used below.
        H = nfaces  # Represents any die higher than the selected dice.
        L = H + 1  # Likewise, but for lower dice.

        # Enumerate the faces by ordinality rather than face value, to
        # preserve the input order and to accommodate non-comparable die
        # values.  For example, this enumerates three six-sided dice
        # numerically from (0, 0, 0) to (5, 5, 5).
        pweight: dict[tuple[_T, ...], Weight] = {}
        for icombo in itertools.combinations_with_replacement(range(nfaces), keep):
            # Get the range of face numbers.
            low = icombo[0]
            high = icombo[-1]
            # How many faces are before low or after high?
            nlow = low
            nhigh = nfaces - 1 - high
            # What are the total weights of the low and high dice?
            weights[L] = sum(weights[i] for i in range(nlow))
            weights[H] = sum(weights[nfaces - 1 - i] for i in range(nhigh))

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
            base_counter = Counter(icombo)
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
                    counts = tuple(sorted(counter.values()))
                    cperms = multiset_perm(counts)
                    # cweight = math.prod(weights[k] ** n for k, n in counter.items())
                    cweight = math.prod(weights[k] for k in counter.elements())
                    weight += cweight * cperms

            # Translate ordinals to face values and base weights.
            vcombo = tuple(faces[p] for p in icombo)
            pweight[vcombo] = weight
        return Dice.from_pairs(pweight.items())

    def sum(self) -> Die[ET_co]:
        """Sum the dice in each roll and return the resulting PMF."""
        weights: dict[ET_co, Weight] = {}
        for combo, count in self.mapping.items():
            total: ET_co
            match combo:
                case (str(), *_):
                    # Use join on strings.
                    total = "".join(combo)  # type: ignore
                case (float(), *_):
                    # Use fsum on floats.
                    total = math.fsum(combo)  # type: ignore
                case _:
                    # Use sum on everything else.
                    total = sum(combo)  # type: ignore
            weights.setdefault(total, 0)
            weights[total] += count
        pairs: Iterable[tuple[ET_co, Weight]] = weights.items()
        return Die.from_pairs(pairs)
