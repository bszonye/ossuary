"""Unit tests for bones.warhammer.Attribute class."""

from bones.warhammer import Attribute


class TestAttributeInit:
    """Test the Attribute class constructor."""

    def test_simple(self):
        """Test with minimal arguments."""
        a = Attribute(int, "Test Attribute")
        assert a.factory == int
        assert a.name == "Test Attribute"
        assert a.python == "test_attribute"
        assert a.toml == "test-attribute"
        assert isinstance(a.aliases, frozenset)
        assert a.aliases == frozenset(
            {
                "Test Attribute",
                "test attribute",
                "test-attribute",
                "test_attribute",
            }
        )
        assert repr(a) == "Attribute(int, 'Test Attribute')"

    def test_custom_python(self):
        """Test python parameter."""
        a = Attribute(str, "Range", python="attack_range")
        assert a.name == "Range"
        assert a.python == "attack_range"
        assert a.toml == "range"
        assert a.aliases == frozenset(
            {
                "Range",
                "attack-range",
                "attack_range",
                "range",
            }
        )

    def test_custom_toml(self):
        """Test toml parameter."""
        a = Attribute(str, "Müllerstraße", toml="muellerstrasse")
        assert a.name == "Müllerstraße"
        assert a.python == "müllerstrasse"
        assert a.toml == "muellerstrasse"
        assert a.aliases == frozenset(
            {
                "Müllerstraße",
                "m-llerstrasse",
                "m_llerstrasse",
                "muellerstrasse",
                "müllerstrasse",
            }
        )

    def test_custom_aliases(self):
        """Test aliases parameter."""
        a = Attribute(float, "ABC", aliases=("Alpha", "Alpha 1 2 3"))
        assert a.aliases == frozenset(
            {
                "ABC",
                "Alpha 1 2 3",
                "Alpha",
                "abc",
                "alpha 1 2 3",
                "alpha",
                "alpha-1-2-3",
                "alpha_1_2_3",
            }
        )
        assert repr(a) == "Attribute(float, 'ABC', aliases=('Alpha', 'Alpha 1 2 3'))"

    def test_codify_tomlize(self):
        """Test codify & tomlize corner cases."""
        a = Attribute(str, "+-123 : grüßen ? ABC-+")
        assert a.name == "+-123 : grüßen ? ABC-+"
        assert a.python == "n123_grüssen_abc_"
        assert a.toml == "-123-gr_ssen-abc-"
