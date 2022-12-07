"""bones.__main__: utility script for dice analysis.

usage: bones [args...]
       python -m bones [args...]

TODO: options, arguments, and usage notes
"""

__all__ = ["plot_demo", "profile_demo", "main"]

import sys
from fractions import Fraction
from typing import Union

import lea
from lea.leaf import dice

Randomizable = Union[int, lea.Lea]


def chain_rolls(n: Randomizable, d: lea.Lea) -> lea.Lea:
    """Analyze one step in the attack sequence."""
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx: lea.Lea = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)


def profile_demo() -> None:
    """Demonstrate profile classes."""


def plot_demo() -> None:
    """Show a few simple placeholder demos."""
    try:
        import matplotlib
        from matplotlib import pyplot
    except ImportError:  # pragma: no cover
        return

    # attacks
    attacks: lea.Lea = dice(1)
    # hit: 4+ exploding 6s
    hit = (0, Fraction(3, 6)), (1, Fraction(2, 6)), (2, Fraction(1, 6))
    # wound: 4+, TODO: 1 MW instead on 6
    wound = (0, Fraction(1, 2)), (1, Fraction(1, 2))
    # rend & damage: TBD
    # rend = 0
    # damage = 1

    hits = chain_rolls(attacks, lea.pmf(hit))
    print("hits")
    print(hits)
    wounds = chain_rolls(hits, lea.pmf(wound))
    print("wounds")
    print(wounds)

    # demo with custom plot
    pyplot.ion()
    domain = hits.support
    ratio = hits.ps
    pyplot.bar(range(len(domain)), ratio, tick_label=domain, align="center")
    pyplot.ylabel("Probability")
    pyplot.title("Hits")
    if sys.__stdout__.isatty():  # pragma: no cover
        pyplot.show(block=True)

    # demo with lea.plot()
    hits.plot(title="Wounds", color="red")
    print("interactive:", pyplot.isinteractive())
    if sys.__stdout__.isatty():  # pragma: no cover
        pyplot.show(block=True)
    print(matplotlib.backends.backend)


def main() -> None:
    """Script entry point. Command-line interface TBD."""
    print(__name__)
    profile_demo()
    plot_demo()


if __name__ == "__main__":
    main()
    sys.exit(0)
