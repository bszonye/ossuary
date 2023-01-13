"""bones.pmf: color utilities."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

__all__ = [
    "ColorTriplet",
    "adjust_lightness",
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
    return 87098 / 409605 * r + 175762 / 245763 * g + 12673 / 175545 * b


def invert(c: ColorTriplet, /) -> ColorTriplet:
    """Invert all channels in an RGB color triplet."""
    r, g, b = c
    return 1.0 - r, 1.0 - g, 1.0 - b


def adjust_lightness(ratio: float, c: ColorTriplet, /) -> ColorTriplet:
    """Increase or decrease L* lightness by a ratio."""
    if ratio != 1.0:
        L0 = lightness(c)
        L1 = ratio * L0
        c = darken(L1, c) if ratio < 1.0 else lighten(L1, c)
    return c


def set_lightness(L1: float, c: ColorTriplet, /) -> ColorTriplet:
    """Set L* lightness to a given value."""
    L0 = lightness(c)
    if L0 != L1:
        c = darken(L1, c) if L1 < L0 else lighten(L1, c)
    return c


def darken(L1: float, c: ColorTriplet, /) -> ColorTriplet:
    """Darken a color from one L* value to another."""
    L1 = max(0, L1)
    L0 = lightness(c)
    if L1 < L0:
        ratio = L1 / L0
        r, g, b = c
        c = ratio * r, ratio * g, ratio * b
    return c


def lighten(L1: float, c: ColorTriplet, /) -> ColorTriplet:
    """Darken a color from one L* value to another."""
    L1 = min(L1, 1.0)
    L0 = lightness(c)
    if L0 == 0.0:  # black into gray
        return L1, L1, L1
    if L0 < L1:
        ratio = L1 / L0
        r, g, b = c
        # Lighten as much as possible without desaturating.
        headroom = 1.0 / max(r, g, b)
        if ratio < headroom:
            return ratio * r, ratio * g, ratio * b
        if 1.0 < headroom:
            c = headroom * r, headroom * g, headroom * b
            L0 = lightness(c)
        # Desaturate to finish lightening.
        ratio = (1 - L1) / (1 - L0)
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
    elif 0.5 < Lstar:
        # Dim bright colors, with a partial exception for yellows.
        yellow = 1.0 - 6.0 * abs(min(hue, 1 / 3) - (1 / 6))
        Lnew = 0.5 + 0.33 * yellow
        r, g, b = darken(Lnew, (r, g, b))
    if t in (tmin, tmax):
        # Emphasize the colors at the end of the range.
        r, g, b = adjust_lightness(highlight, (r, g, b))
    return r, g, b
