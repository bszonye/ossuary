"""Unit tests for bones.warhammer.Attribute class."""

from bones.warhammer import Attribute


def test_attribute_init():
    """Test constructor options."""
    # TODO: Don't just print stuff, test it!
    print()
    a1 = Attribute(int, "Test Attribute")
    print(a1)
    a2 = Attribute(int, "Test Attribute", python="test2")
    print(a2)
    a3 = Attribute(int, "Test Attribute", aliases=("foo", "bar", "test-attribute"))
    print(a3)
    print(a3.aliases)
    print(Attribute(int, "Test-Attribute"))
    print(Attribute(int, "Test_Attribute"))
    print(Attribute(int, "69 dude"))
    print(Attribute(int, "__init__"))
    print(Attribute(int, "test", toml="-test-"))
