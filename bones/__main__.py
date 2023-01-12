"""bones.__main__: utility script for dice analysis.

usage: bones [args...]
       python -m bones [args...]

TODO: options, arguments, and usage notes
"""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = ["plot_demo", "main"]

import sys

from .pmf import PMF


def plot_demo() -> None:
    """Show placeholder demos."""
    if sys.__stdout__.isatty():  # pragma: no cover
        n = int(sys.argv[1]) if 1 < len(sys.argv) else 10
        PMF(range(1, 1 + n)).plot(q=n)
        spectrum = {x: x for x in range(1, 31)}
        PMF(spectrum)  # .plot()


def main() -> None:
    """Script entry point. Command-line interface TBD."""
    plot_demo()


if __name__ == "__main__":
    main()
    sys.exit(0)
