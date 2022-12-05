"""Unit tests for bones.warhammer.Attribute class."""

import pytest

from bones.warhammer import Attribute


class TestAttributeInit:
    """Test the Attribute class constructor."""

    def test_simple(self) -> None:
        """Test with minimal arguments."""
        attr = Attribute("Simple Attribute")
        assert attr.name == "Simple Attribute"
        assert attr.factory == str

    def test_factory_arg(self) -> None:
        """Test with factory argument."""
        attr = Attribute("Factory Positional", int)
        assert attr.name == "Factory Positional"
        assert attr.factory == int

    def test_factory_keyword(self) -> None:
        """Test with factory keyword argument."""
        attr = Attribute("Factory Keyword", factory=float)
        assert attr.name == "Factory Keyword"
        assert attr.factory == float

    def test_all_keywords(self) -> None:
        """Test with all keyword arguments."""
        attr = Attribute(factory=dict, name="All Keywords")
        assert attr.name == "All Keywords"
        assert attr.factory == dict


class TestAttributeNormalize:
    """Test the Attribute.normalize method."""

    normalize_table = {
        # No transformation required.
        "abc": ("abc", "abc", "abc", "abc", "abc"),
        # Case folding.
        "ABC": ("abc", "abc", "abc", "abc", "abc"),
        "Große": ("grosse", "grosse", "grosse", "grosse", "grosse"),
        # Connect words.
        "a b": ("a_b", "a-b", "a:b", "ab", "a_b"),
        "a b c": ("a_b_c", "a-b-c", "a:b:c", "abc", "a_b_c"),
        "a b c d": ("a_b_c_d", "a-b-c-d", "a:b:c:d", "abcd", "a_b_c_d"),
        # Replace non-identifier characters with connectors.
        "abc-123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "abc/123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "abc:123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        # Replace existing connectors (Unicode category Pc).
        "abc_123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "abc‿123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "abc⁀123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "abc⁔123": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        # Remove redundant connectors.
        "  abc  123  ": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "+-abc-+-123-+": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        "==abc===123==": ("abc_123", "abc-123", "abc:123", "abc123", "abc_123"),
        # Python rules: first character is not xid_start.
        "123 abc": ("A123_abc", "123-abc", "123:abc", "A123abc", "123_abc"),
        "[123-abc]": ("A123_abc", "123-abc", "123:abc", "A123abc", "123_abc"),
        "__123_abc__": ("A123_abc", "123-abc", "123:abc", "A123abc", "123_abc"),
        # Python rules: name is a builtin or keyword.
        "None": ("none", "none", "none", "none", "none"),  # ok because of casefold!
        "break": ("break_", "break", "break", "break_", "break"),
        "range": ("range_", "range", "range", "range_", "range"),
    }

    @pytest.mark.parametrize("name", normalize_table)
    def test_normalize(self, name: str) -> None:
        """Test Attribute.normalize method with minimal arguments."""
        python, toml, colon, null, nopython = self.normalize_table[name]
        assert python == Attribute.normalize(name)
        assert python == Attribute.normalize(name, connector="_")
        assert toml == Attribute.normalize(name, connector="-")
        assert colon == Attribute.normalize(name, connector=":")
        assert null == Attribute.normalize(name, connector="")
        assert nopython == Attribute.normalize(name, python=False)

    def test_normalize_error(self) -> None:
        """Test Attribute.normalize with incompatible parameters."""
        with pytest.raises(ValueError):
            Attribute.normalize("", connector=":", python=True)

    @pytest.mark.parametrize("name", normalize_table)
    def test_python_toml(self, name: str) -> None:
        """Test Attribute.python and .toml methods."""
        python, toml, *_ = self.normalize_table[name]
        attr = Attribute(name)
        assert python == attr.python()
        assert toml == attr.toml()


class TestAttributeRepr:
    """Test Attribute.__repr__ special method."""

    def test_simple(self) -> None:
        """Test with minimal arguments."""
        attr = Attribute("Simple Attribute")
        out = "Attribute('Simple Attribute')"
        assert repr(attr) == out

    def test_factory_redundant(self) -> None:
        """Test with minimal arguments."""
        attr = Attribute("Factory Redundant", str)
        out = "Attribute('Factory Redundant')"
        assert repr(attr) == out

    def test_factory_arg(self) -> None:
        """Test with factory argument."""
        attr = Attribute("Factory Positional", int)
        out = "Attribute('Factory Positional', int)"
        assert repr(attr) == out

    def test_factory_keyword(self) -> None:
        """Test with factory keyword argument."""
        attr = Attribute("Factory Keyword", factory=float)
        out = "Attribute('Factory Keyword', float)"
        assert repr(attr) == out

    def test_all_keywords(self) -> None:
        """Test with all keyword arguments."""
        attr = Attribute(factory=dict, name="All Keywords")
        out = "Attribute('All Keywords', dict)"
        assert repr(attr) == out
