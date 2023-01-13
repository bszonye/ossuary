"""bones.pmf: color utilities."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import colorsys
from typing import TypeAlias

ColorTriplet: TypeAlias = tuple[float, float, float]


def invert(c: ColorTriplet, /) -> ColorTriplet:
    """Invert all channels in an RGB color triplet."""
    r, g, b = c
    return 1.0 - r, 1.0 - g, 1.0 - b


def darken(L1: float, c: ColorTriplet, /) -> ColorTriplet:
    """Darken a color from one L* value to another."""
    L0 = lightness(c)
    if L1 < L0:
        r, g, b = c
        ratio = L1 / L0
        c = ratio * r, ratio * g, ratio * b
    return c


def lighten(L1: float, c: ColorTriplet, /) -> ColorTriplet:
    """Darken a color from one L* value to another."""
    L0 = lightness(c)
    if L0 < L1:
        r, g, b = invert(c)
        ratio = (1 - L1) / (1 - L0)
        c = invert((ratio * r, ratio * g, ratio * b))
    return c


def lightness(c: ColorTriplet, /) -> float:
    """Convert an sRGB triplet to L*a*b* lightness."""
    r, g, b = c
    return 87098 / 409605 * r + 175762 / 245763 * g + 12673 / 175545 * b


def plot_color(
    p: float,
    /,
    *,
    pmin: float = 0.0,
    pmax: float = 1.0,
    cmin: float = 0.75,
    cmax: float = 0.0,
    outliers: bool = False,
) -> ColorTriplet:
    """Translate a range of numbers to a range of colors."""
    t = float((p - pmin) / (pmax - pmin)) if pmax != pmin else 0.5
    hue = (cmax - cmin) * t + cmin
    # Widen CMY and narrow RGB to smooth color transitions.
    # hue += math.sin(6.0 * math.pi * hue) / 36.0
    hue %= 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    Lstar = lightness((r, g, b))
    if Lstar < 0.5:
        # Lighten dark colors.
        Lnew = Lstar + (0.5 - Lstar) * 0.4
        r, g, b = lighten(Lnew, (r, g, b))
    elif 0.5 < Lstar:
        # Dim bright colors, with a partial exception for yellows.
        yellow = 1.0 - 6.0 * abs(min(hue, 1 / 3) - (1 / 6))
        Lnew = 0.5 + 0.33 * yellow
        r, g, b = darken(Lnew, (r, g, b))
    if outliers and p in (pmin, pmax):
        # Emphasize the colors at the end of the range.
        ratio = 0.65
        r, g, b = ratio * r, ratio * g, ratio * b
    return r, g, b
