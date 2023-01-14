"""Unit tests for the bones.pmf module."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import colorsys

from pytest import approx  # pyright: ignore[reportUnknownVariableType]

from bones.color import (
    adjust_lightness,
    clip,
    ColorTriplet,
    darken,
    interpolate_color,
    invert,
    lighten,
    lightness,
    set_lightness,
)


def hue(c: ColorTriplet) -> float:
    return colorsys.rgb_to_hsv(*c)[0]


def saturation(c: ColorTriplet) -> float:
    return colorsys.rgb_to_hsv(*c)[1]


class TestLightness:
    def test_gray(self) -> None:
        assert lightness((0.0, 0.0, 0.0)) == 0.0
        assert lightness((0.5, 0.5, 0.5)) == approx(0.5)
        assert lightness((1.0, 1.0, 1.0)) == approx(1.0)
        assert lightness((1.0, 1.0, 1.0)) <= 1.0

    def test_red(self) -> None:
        assert lightness((1.0, 0.0, 0.0)) == approx(87098 / 409605)

    def test_green(self) -> None:
        assert lightness((0.0, 1.0, 0.0)) == approx(175762 / 245763)

    def test_blue(self) -> None:
        assert lightness((0.0, 0.0, 1.0)) == approx(12673 / 175545)


class TestClip:
    def test_over(self) -> None:
        assert clip((1.1, 1.1, 1.1)) == (1.0, 1.0, 1.0)

    def test_under(self) -> None:
        assert clip((-0.1, -0.1, -0.1)) == (0.0, 0.0, 0.0)


class TestInvert:
    def test_gray(self) -> None:
        assert invert((0.0, 0.0, 0.0)) == (1.0, 1.0, 1.0)
        assert invert((0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5)
        assert invert((0.75, 0.75, 0.75)) == (0.25, 0.25, 0.25)
        assert invert((0.9, 0.9, 0.9)) == approx((0.1, 0.1, 0.1))
        assert invert((1.0, 1.0, 1.0)) == (0.0, 0.0, 0.0)

    def test_red(self) -> None:
        assert invert((1.0, 0.0, 0.0)) == (0.0, 1.0, 1.0)
        assert invert((0.5, 0.0, 0.0)) == (0.5, 1.0, 1.0)

    def test_green(self) -> None:
        assert invert((0.0, 1.0, 0.0)) == (1.0, 0.0, 1.0)
        assert invert((0.0, 0.5, 0.0)) == (1.0, 0.5, 1.0)

    def test_blue(self) -> None:
        assert invert((0.0, 0.0, 1.0)) == (1.0, 1.0, 0.0)
        assert invert((0.0, 0.0, 0.5)) == (1.0, 1.0, 0.5)


class TestAdjustLightness:
    def test_same(self) -> None:
        assert adjust_lightness(1.0, (1.0, 0.5, 0.0)) == (1.0, 0.5, 0.0)

    def test_darken(self) -> None:
        assert adjust_lightness(0.5, (1.0, 0.5, 0.0)) == (0.5, 0.25, 0.0)

    def test_lighten(self) -> None:
        assert adjust_lightness(2.0, (0.5, 0.25, 0.0)) == (1.0, 0.5, 0.0)


class TestSetLightness:
    def test_same(self) -> None:
        assert set_lightness(0.0, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)

    def test_darken(self) -> None:
        assert set_lightness(0.25, (0.5, 0.5, 0.5)) == approx((0.25, 0.25, 0.25))

    def test_lighten(self) -> None:
        assert set_lightness(0.5, (0.25, 0.25, 0.25)) == approx((0.5, 0.5, 0.5))


class TestDarken:
    def test_same(self) -> None:
        assert darken(0.0, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
        assert darken(1.0, (0.5, 0.5, 0.5)) == approx((0.5, 0.5, 0.5))

    def test_darken(self) -> None:
        c = 1.0, 0.5, 0.0
        assert hue(darken(0.5, c)) == approx(hue(c))
        assert darken(0.25, (0.5, 0.5, 0.5)) == approx((0.25, 0.25, 0.25))


class TestLighten:
    def test_same(self) -> None:
        assert lighten(0.0, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
        assert lighten(0.0, (0.5, 0.5, 0.5)) == approx((0.5, 0.5, 0.5))

    def test_brighten(self) -> None:
        c = 0.0, 0.25, 0.5
        assert hue(lighten(0.5, c)) == approx(hue(c))
        assert lighten(0.5, (0.25, 0.25, 0.25)) == approx((0.5, 0.5, 0.5))

    def test_desaturate(self) -> None:
        # no headroom
        orange1a = (1.0, 0.5, 0.0)
        orange1b = lighten(0.75, orange1a)
        assert lightness(orange1b) == approx(0.75)
        assert hue(orange1a) == approx(hue(orange1a))
        # with headroom
        orange2a = (0.75, 0.5, 0.0)
        orange2b = lighten(0.75, orange2a)
        assert lightness(orange2b) == approx(0.75)
        assert hue(orange2a) == approx(hue(orange2a))


class TestInterpolateColor:
    def test_zero_interval(self) -> None:
        red = interpolate_color(0.0, tmax=0.0)
        assert hue(red) == approx(0.0)
        green = interpolate_color(0.0, tmax=0.0, hmax=0.3333333)
        assert hue(green) == approx(0.3333333)
        assert saturation(green) == approx(1.0)
        assert lightness(green) == approx(0.5)
        blue = interpolate_color(0.0, tmax=0.0, hmax=0.6666667)
        assert hue(blue) == approx(0.6666667)

    def test_highlight(self) -> None:
        # Use a bright color that gets dimmed to exactly 0.5 lightness.
        cyan1 = interpolate_color(0.0, tmax=0.0, hmax=0.5, highlight=0.5)  # darker
        assert hue(cyan1) == approx(0.5)
        assert saturation(cyan1) == approx(1.0)
        assert lightness(cyan1) == approx(0.25)
        cyan2 = interpolate_color(0.0, tmax=0.0, hmax=0.5, highlight=1.5)  # lighter
        assert hue(cyan2) == approx(0.5)
        assert saturation(cyan2) == approx(1.0)
        assert lightness(cyan2) == approx(0.75)

    def test_lerp(self) -> None:
        # Default ranges.
        assert hue(interpolate_color(0.0)) == approx(0.75)
        assert hue(interpolate_color(0.5)) == approx(0.375)
        assert hue(interpolate_color(1.0)) == approx(0.0)
        # Matching ranges.
        assert hue(interpolate_color(0.0, hmin=0.0, hmax=1.0)) == approx(0.0)
        assert hue(interpolate_color(0.25, hmin=0.0, hmax=1.0)) == approx(0.25)
        assert hue(interpolate_color(0.5, hmin=0.0, hmax=1.0)) == approx(0.5)
        assert hue(interpolate_color(0.75, hmin=0.0, hmax=1.0)) == approx(0.75)
