"""bones.warhammer: dice analysis for Warhammer games."""

__all__ = (
    "AttackCounter",
    "Characteristic",
    "Datasheet",
    "Profile",
    "Unit",
    "Warscroll",
    "Weapon",
    "chain_rolls",
)

import dataclasses
import sys
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, IO, Optional, SupportsFloat, Type, TypeVar, Union

import lea

# Conditionally import tomllib (Python 3.11+) or tomli.
# (mypy doesn't understand the try/except idiom)
if sys.version_info[:2] < (3, 11):
    import tomli as tomllib
else:
    import tomllib

# Type definitions.
# TODO: remove unused types
Numeric = SupportsFloat  # reasonable approximation of "numeric"
Distance = Numeric  # e.g. for move & range characteristics
ProfileType = TypeVar("ProfileType", bound="Profile")
Randomizable = Union[Numeric, lea.Lea]
NameDict = dict[str, Any]

lea.set_prob_type("r")


@dataclasses.dataclass(frozen=True)
class AttackCounter:
    """Results counter for each step of the attack sequence."""

    # TODO: addition operator?
    attacks: int
    wounds: int = 0
    mortals: int = 0
    kills: int = 0


@dataclasses.dataclass(frozen=True)
class Characteristic:
    """General unit & weapon characteristics."""

    name: str
    value: Any  # TODO: refine this? depends on the characteristic type

    def __iter__(self) -> Iterator[Any]:
        """Enable conversion to (name, value) for dict construction."""
        return iter(dataclasses.astuple(self))


class Profile(Mapping[str, Characteristic]):
    """Tabular data for units and weapons.

    Most Warhammer stats are **characteristics** organized into tables
    with a named **profile** in each row and the characteristic values
    in columns.  This base class represents basic profile data with
    subscript notation for accessing the characteristics by name.

    """

    name: Optional[str]
    characteristics: dict[str, Characteristic]

    def __init__(
        self,
        __c: Union[NameDict, Iterable[Any]],
        /,
        *,
        name: Optional[str] = None,
    ):
        """Initialize characteristics."""
        # Convert input data to a characteristic dictionary.
        # TODO: Handle specialized values (e.g. target numbers).
        items = __c.items() if isinstance(__c, Mapping) else __c
        characteristics = (
            item if isinstance(item, Characteristic) else Characteristic(*item)
            for item in items
        )
        self.characteristics = {item.name: item for item in characteristics}
        self.name = name

    @classmethod
    def load(
        cls: Type[ProfileType], __fp: IO[bytes], /, *args: Any, **kwargs: Any
    ) -> ProfileType:
        """Construct a new Profile object from a TOML file."""
        data = tomllib.load(__fp)
        return cls(data, *args, **kwargs)

    @classmethod
    def loads(
        cls: Type[ProfileType], __s: str, /, *args: Any, **kwargs: Any
    ) -> ProfileType:
        """Construct a new Profile object from a TOML string."""
        data = tomllib.loads(__s)
        return cls(data, *args, **kwargs)

    def __getitem__(self, key: str) -> Characteristic:
        """Subscript notation to simplify access to characteristics."""
        return self.characteristics[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of all characteristics."""
        return iter(self.characteristics)

    def __len__(self) -> int:
        """Report the number of characteristics."""
        return len(self.characteristics)

    def __repr__(self) -> str:
        """Dump object data in repr() format."""
        data = tuple(c for c in self.characteristics.values())
        return "{}({}, name={})".format(
            type(self).__name__,
            repr(data),
            repr(self.name),
        )


class Weapon(Profile):
    """Weapon profile."""


class Warscroll(Profile):
    """Unit characteristics & abilities for Warhammer Age of Sigmar."""


class Datasheet(Profile):
    """Unit characteristics & abilities for Warhammer 40,000."""


class Unit:
    """Fully specified army unit.

    Includes the unit profile (warscroll or datasheet), number of
    models, and all optional selections such as weapon options or
    upgrades.
    """


def chain_rolls(n: Randomizable, d: lea.Lea) -> lea.Lea:
    """Analyze one step in the attack sequence."""
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
