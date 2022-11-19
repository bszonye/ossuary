import dataclasses
from typing import NewType, Optional
from typing import SupportsFloat as Numeric

import lea

__all__ = (
    "AttackCounter",
    "Datasheet",
    "Warscroll",
    "Weapon",
    "chain_rolls",
)

# TODO: set up a type checker
Distance = NewType("Distance", Numeric)  # e.g. for move & range characteristics
TargetNumber = NewType("TargetNumber", int)  # target for roll characteristics
DiceModifier = NewType("DiceModifier", int)  # e.g. rend

lea.set_prob_type("r")


@dataclasses.dataclass(frozen=True)
class AttackCounter:
    # TODO: addition operator?
    attacks: int
    wounds: int = 0
    mortals: int = 0
    kills: int = 0


class Weapon:
    # TODO: revisit names, rename range
    range: Optional[Distance] = None
    hit: Optional[TargetNumber] = None
    wound: Optional[TargetNumber] = None
    rend: Optional[DiceModifier] = None
    pass


class Warscroll:
    # Warhammer Age of Sigmar unit characteristics & abilities
    pass


class Datasheet:
    # Warhammer 40,000 unit characteristics & abilities
    pass


def chain_rolls(n, d):
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
