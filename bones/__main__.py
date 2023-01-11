"""bones.__main__: utility script for dice analysis.

usage: bones [args...]
       python -m bones [args...]

TODO: options, arguments, and usage notes
"""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = ["plot_demo", "main"]

import sys

from .roll import d6


def plot_demo() -> None:
    """Show placeholder demos."""
    try:
        import matplotlib as mpl
    except ImportError:  # pragma: no cover
        return

    if sys.__stdout__.isatty():  # pragma: no cover
        (3 @ d6).plot()
        print(mpl.backends.backend)


def main() -> None:
    """Script entry point. Command-line interface TBD."""
    plot_demo()


if __name__ == "__main__":
    main()
    sys.exit(0)
