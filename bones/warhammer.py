"""bones.warhammer: dice analysis for Warhammer games."""

__all__ = (
    "AttackCounter",
    "Datasheet",
    "Warscroll",
    "Weapon",
    "chain_rolls",
)

import dataclasses
from typing import NewType, Optional, SupportsFloat as Numeric, Union

import lea

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


class Weapon:
    """Weapon charcteristics."""

    # TODO: rename range
    range: Optional[Distance] = None
    to_hit: Optional[TargetNumber] = None
    to_wound: Optional[TargetNumber] = None
    rend: Optional[DiceModifier] = None
    pass


class Warscroll:
    """Unit characteristics & abilities for Warhammer Age of Sigmar."""

    pass


class Datasheet:
    """Unit characteristics & abilities for Warhammer 40,000."""

    pass


def chain_rolls(n: Randomizable, d: lea.Lea) -> lea.Lea:
    """Analyze one step in the attack sequence."""
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
