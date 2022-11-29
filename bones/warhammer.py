"""bones.warhammer: dice analysis for Warhammer games."""

__all__ = (
    "AttackCounter",
    "Datasheet",
    "Warscroll",
    "Weapon",
    "chain_rolls",
)

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NewType, SupportsFloat as Numeric, Union

import lea

# TODO: remove unused types
Distance = Numeric  # e.g. for move & range characteristics
TargetNumber = NewType("TargetNumber", int)  # target for roll characteristics
DiceModifier = NewType("DiceModifier", int)  # e.g. rend
Randomizable = Union[Numeric, lea.Lea]

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


class Profile(Mapping[str, Characteristic]):
    """Tabular data for units and weapons.

    Most Warhammer stats are **characteristics** organized into tables
    with a named **profile** in each row and the characteristic values
    in columns.  This base class represents basic profile data with
    subscript notation for accessing the characteristics by name.
    """

    name: str
    characteristics: dict[str, Characteristic]

    def __init__(self, name: str, characteristics: Iterable[Characteristic]) -> None:
        """Initialize characteristics."""
        self.characteristics = {item.name: item for item in characteristics}

    def __getitem__(self, key: str) -> Characteristic:
        """Subscript notation to simplify access to characteristics."""
        return self.characteristics[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.characteristics)

    def __len__(self) -> int:
        return len(self.characteristics)


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
