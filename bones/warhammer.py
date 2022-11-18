import dataclasses

import lea

__all__ = (
    "AttackCounter",
    "DefenseProfile",
    "Warscroll",
    "WeaponProfile",
    "chain_rolls",
)

lea.set_prob_type("r")


@dataclasses.dataclass(frozen=True)
class AttackCounter:
    # TODO: addition operator?
    attacks: int
    wounds: int = 0
    mortals: int = 0


class DefenseProfile:
    pass


class WeaponProfile:
    pass


class Warscroll:
    pass


def chain_rolls(n, d):
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
