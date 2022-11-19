from fractions import Fraction

import lea
from lea.leaf import dice

import matplotlib
from matplotlib import pyplot

from .warhammer import chain_rolls


def demo():
    # attacks
    attacks = dice(1)
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
    pyplot.show(block=True)

    # demo with lea.plot()
    hits.plot(title="Wounds", color="red")
    print("interactive:", pyplot.isinteractive())
    pyplot.show(block=True)
    print(matplotlib.backends.backend)


def main():
    demo()


if __name__ == "__main__":
    main()
