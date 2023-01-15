"""bones.__main__: utility script for dice analysis.

usage: bones [args...]
       python -m bones [args...]

TODO: options, arguments, and usage notes
"""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = ["main"]

import sys
from collections.abc import Iterable, Sequence
from importlib import import_module
from types import EllipsisType

from .pmf import PMF


def eval_demo(
    expressions: Iterable[str], *, interactive: bool | EllipsisType = Ellipsis
) -> None:
    """Evaluate expressions from the command line."""
    from . import __all__ as exports, __name__ as package_name

    package = import_module(package_name)
    bones_modules = ["color", "pmf", "roll", "warhammer"]
    stdlib_modules = ["math", "operator"]

    g = {package_name: package}
    g |= {name: import_module(f".{name}", package_name) for name in bones_modules}
    g |= {name: import_module(name) for name in stdlib_modules}
    g |= {name: getattr(package, name) for name in exports}

    if interactive is Ellipsis:  # autodetect based on isatty
        interactive = sys.__stdout__.isatty()

    for expression in expressions:
        v = eval(expression, g)
        if isinstance(v, PMF):
            if interactive:  # pragma: no cover
                v.plot()
            else:
                print("\n".join(v.tabulate(":.2%")))
        elif v is not None:
            print(v)


def main(argv: Sequence[str] | None = None) -> None:
    """Script entry point. Command-line interface TBD."""
    if argv is None:  # pragma: no cover
        argv = sys.argv
    eval_demo(argv[1:])


if __name__ == "__main__":
    main()
    sys.exit(0)
