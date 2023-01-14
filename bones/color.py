"""bones.pmf: color utilities."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "adjust_lightness",
    "clip",
    "ColorTriplet",
    "darken",
    "interpolate_color",
    "invert",
    "lighten",
    "lightness",
    "set_lightness",
]

import colorsys
from typing import TypeAlias

ColorTriplet: TypeAlias = tuple[float, float, float]


def lightness(c: ColorTriplet, /) -> float:
    """Convert an sRGB triplet to L*a*b* lightness."""
    r, g, b = c
    lstar = 87098 / 409605 * r + 175762 / 245763 * g + 12673 / 175545 * b
    return min(lstar, 1.0)  # color channels need to stay in the [0, 1] interval


def clip(c: ColorTriplet, /) -> ColorTriplet:
    """Clip color channels to the [0, 1] interval."""
    r, g, b = c
    r = max(0.0, min(r, 1.0))
    g = max(0.0, min(r, 1.0))
    b = max(0.0, min(r, 1.0))
    return r, g, b


def invert(c: ColorTriplet, /) -> ColorTriplet:
    """Invert all channels in an RGB color triplet."""
    r, g, b = c
    return 1.0 - r, 1.0 - g, 1.0 - b


def adjust_lightness(ratio: float, c: ColorTriplet, /) -> ColorTriplet:
    """Increase or decrease L* lightness by a ratio."""
    if ratio != 1.0:
        lstar0 = lightness(c)
        lstar = ratio * lstar0
        c = darken(lstar, c) if ratio < 1.0 else lighten(lstar, c)
    return c


def set_lightness(lstar: float, c: ColorTriplet, /) -> ColorTriplet:
    """Change color to match a given L* lightness."""
    lstar0 = lightness(c)
    if lstar0 != lstar:
        c = darken(lstar, c) if lstar < lstar0 else lighten(lstar, c)
    return c


def darken(lstar: float, c: ColorTriplet, /) -> ColorTriplet:
    """Darken a color from one L* value to another."""
    lstar = max(0, lstar)
    lstar0 = lightness(c)
    if lstar < lstar0:
        ratio = lstar / lstar0
        r, g, b = c
        c = ratio * r, ratio * g, ratio * b
    return c


def lighten(lstar: float, c: ColorTriplet, /) -> ColorTriplet:
    """Lighten a color from one L* value to another."""
    lstar = min(lstar, 1.0)
    lstar0 = lightness(c)
    if lstar0 == 0.0:  # black into gray
        return lstar, lstar, lstar
    if lstar0 < lstar:
        ratio = lstar / lstar0
        r, g, b = c
        # Lighten as much as possible without desaturating.
        headroom = 1.0 / max(r, g, b)
        if ratio < headroom:
            return ratio * r, ratio * g, ratio * b
        if 1.0 < headroom:
            c = headroom * r, headroom * g, headroom * b
            lstar0 = lightness(c)
        # Desaturate to finish lightening.
        ratio = (1 - lstar) / (1 - lstar0)
        r, g, b = invert(c)
        c = invert((ratio * r, ratio * g, ratio * b))
    return c


def interpolate_color(
    t: float,
    /,
    *,
    tmin: float = 0.0,
    tmax: float = 1.0,
    hmin: float = 0.75,
    hmax: float = 0.0,
    highlight: float = 1.0,
) -> ColorTriplet:
    """Translate a range of numbers to a range of colors."""
    # Normalize t to the [0.0, 1.0] interval and interpolate a hue.
    x = float((t - tmin) / (tmax - tmin)) if tmax != tmin else 1.0
    hue = ((hmax - hmin) * x + hmin) % 1.0
    # Widen CMY and narrow RGB to smooth color transitions.
    # hue += math.sin(6.0 * math.pi * hue) / 36.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    Lstar = lightness((r, g, b))
    if Lstar < 0.5:
        # Lighten dark colors.
        Lnew = Lstar + (0.5 - Lstar) * 0.4
        r, g, b = lighten(Lnew, (r, g, b))
    else:
        # Dim bright colors, with a partial exception for yellows.
        yellow = 1.0 - 6.0 * abs(min(hue, 1 / 3) - (1 / 6))
        Lnew = 0.5 + 0.33 * yellow
        r, g, b = darken(Lnew, (r, g, b))
    if t in (tmin, tmax):
        # Emphasize the colors at the end of the range.
        r, g, b = adjust_lightness(highlight, (r, g, b))
    return r, g, b
