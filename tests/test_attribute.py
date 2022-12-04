"""Unit tests for bones.warhammer.Attribute class."""

from bones.warhammer import Attribute


class TestAttributeInit:
    """Test the Attribute class constructor."""

    def test_simple(self):
        """Test with minimal arguments."""
        attr = Attribute(int, "Test Attribute")
        assert attr.factory == int
        assert attr.name == "Test Attribute"
        assert attr.python == "test_attribute"
        assert attr.toml == "test-attribute"
        assert isinstance(attr.aliases, frozenset)
        assert attr.aliases == frozenset(
            {
                "Test Attribute",
                "test attribute",
                "test-attribute",
                "test_attribute",
            }
        )

    def test_custom_python(self):
        """Test python parameter."""
        attr = Attribute(str, "Range", python="attack_range")
        assert attr.factory == str
        assert attr.name == "Range"
        assert attr.python == "attack_range"
        assert attr.toml == "range"
        assert attr.aliases == frozenset(
            {
                "Range",
                "attack-range",
                "attack_range",
                "range",
            }
        )

    def test_custom_toml(self):
        """Test toml parameter."""
        attr = Attribute(str, "Müllerstraße", toml="muellerstrasse")
        assert attr.factory == str
        assert attr.name == "Müllerstraße"
        assert attr.python == "müllerstrasse"
        assert attr.toml == "muellerstrasse"
        assert attr.aliases == frozenset(
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
        attr = Attribute(float, "ABC", aliases=("Alpha", "Alpha 1 2 3"))
        assert attr.factory == float
        assert attr.name == "ABC"
        assert attr.python == "abc"
        assert attr.toml == "abc"
        assert attr.aliases == frozenset(
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

    def test_custom_everything(self):
        """Test with all optional parameters."""
        attr = Attribute(int, "A", python="b", toml="c", aliases=("D", "E"))
        assert attr.factory == int
        assert attr.name == "A"
        assert attr.python == "b"
        assert attr.toml == "c"
        assert attr.aliases == frozenset({"A", "D", "E", "a", "b", "c", "d", "e"})


class TestAttributeCodify:
    """Test the Attribute.codify class method."""

    def loop_tests(self, *tests):
        """Loop over several test cases."""
        for given, expected in tests:
            assert Attribute.codify(given) == expected

    def test_casefold(self):
        """Test case folding conversion."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("ABC", "abc"),  # lowercase
            ("große", "grosse"),  # eszett => ss
        )

    def test_connectors(self):
        """Test connector conversion."""
        self.loop_tests(
            ("abcdef", "abcdef"),  # unchanged
            ("abc def", "abc_def"),  # non-alphunumeric => connector
            ("abc : def", "abc_def"),  # multiple characters => single connector
            ("abc-def", "abc_def"),  # hyphen vs underscore
            ("abc_def", "abc_def"),  # hyphen vs underscore
        )

    def test_trim(self):
        """Test strings with conversions at the beginning & end."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("+abc", "abc"),  # avoid leading underscores in Python
            ("abc+", "abc_"),  # keep trailing underscores
            ("+abc+", "abc_"),  # both ends
            ("+/-abc-/+", "abc_"),  # trim extras
        )

    def test_leading_digits(self):
        """Test strings with leading digits."""
        self.loop_tests(
            ("abc123", "abc123"),  # unchanged
            ("123abc", "n123abc"),  # guard leading digits in Python
            ("__123abc__", "n123abc_"),  # guard uncovered digits
        )

    def test_unicode(self):
        """Test strings with non-ASCII characters."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("äbc", "äbc"),  # umlaut => OK in Python
            ("große", "grosse"),  # eszett => ss
            ("Müllerstraße", "müllerstrasse"),  # eszett => ss
            ("++ää++öö++üü++", "ää_öö_üü_"),  # consecutive conversions
        )


class TestAttributeTomlize:
    """Test the Attribute.tomlize codify class method."""

    def loop_tests(self, *tests):
        """Loop over several test cases."""
        for given, expected in tests:
            assert Attribute.tomlize(given) == expected

    def test_casefold(self):
        """Test case folding conversion."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("ABC", "abc"),  # lowercase
            ("große", "grosse"),  # eszett => ss
        )

    def test_connectors(self):
        """Test connector conversion."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("abc def", "abc-def"),  # non-alphunumeric => connector
            ("abc : def", "abc-def"),  # multiple characters => single connector
            ("abc-def", "abc-def"),  # hyphen vs underscore
            ("abc_def", "abc-def"),  # hyphen vs underscore
        )

    def test_trim(self):
        """Test strings with conversions at the beginning & end."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("+abc", "-abc"),  # keep leading hyphens in TOML
            ("abc+", "abc-"),  # keep trailing hypens
            ("+abc+", "-abc-"),  # both ends
            ("+/-abc-/+", "-abc-"),  # trim extras
        )

    def test_leading_digits(self):
        """Test strings with leading digits."""
        self.loop_tests(
            ("abc123", "abc123"),  # unchanged
            ("123abc", "123abc"),  # leading digits are OK in TOML
            ("__123abc__", "-123abc-"),
        )

    def test_unicode(self):
        """Test strings with non-ASCII characters."""
        self.loop_tests(
            ("abc", "abc"),  # unchanged
            ("äbc", "_bc"),  # umlaut => underscore placeholder in TOML
            ("große", "grosse"),  # eszett => ss
            ("Müllerstraße", "m_llerstrasse"),
            ("++ää++öö++üü++", "-__-__-__-"),  # consecutive conversions
        )


class TestAttributeAutoalias:
    """Test the Attribute.autoalias class method."""

    def test_simple(self):
        """Test a simple closure (complete in one pass)."""
        assert Attribute.autoalias("Simple Closure") == frozenset(
            {
                "Simple Closure",
                "simple closure",
                "simple-closure",
                "simple_closure",
            }
        )

    def test_deep(self):
        """Test a deep closure (requires more than one pass)."""
        assert Attribute.autoalias("+-Ä123-Ää Ööß-Üü-+") == frozenset(
            {
                "+-Ä123-Ää Ööß-Üü-+",
                "+-ä123-ää ööss-üü-+",
                "-123-ss-",
                "-_123-__-__ss-__-",
                "_123-__-__ss-__-",
                "n123-ss-",
                "n123_ss_",
                "ä123_ää_ööss_üü_",
            }
        )


class TestAttributeRepr:
    """Test the Attribute.__repr__ special method."""

    def test_simple(self):
        """Test with minimal arguments."""
        attr = Attribute(int, "Test Attribute")
        out = "Attribute(int, 'Test Attribute')"
        assert repr(attr) == out

    def test_custom_python(self):
        """Test python parameter."""
        attr = Attribute(str, "Range", python="attack_range")
        out = "Attribute(str, 'Range', python='attack_range')"
        assert repr(attr) == out

    def test_custom_toml(self):
        """Test toml parameter."""
        attr = Attribute(str, "Müllerstraße", toml="muellerstrasse")
        out = "Attribute(str, 'Müllerstraße', toml='muellerstrasse')"
        assert repr(attr) == out

    def test_custom_aliases(self):
        """Test aliases parameter."""
        attr = Attribute(float, "ABC", aliases=("Alpha", "Alpha 1 2 3"))
        out = "Attribute(float, 'ABC', aliases=('Alpha', 'Alpha 1 2 3'))"
        assert repr(attr) == out

    def test_custom_everything(self):
        """Test with all optional parameters."""
        attr = Attribute(int, "A", python="b", toml="c", aliases=("D", "E"))
        out = "Attribute(int, 'A', python='b', toml='c', aliases=('D', 'E'))"
        assert repr(attr) == out


class TestAttributeComparison:
    """Test comparison operators over Attribute objects."""

    attributes = [  # Sorted for testing.
        Attribute(int, "A", python="b", toml="c", aliases=("D", "E")),
        Attribute(str, "ABC", aliases=("Alpha", "Alpha 1 2 3")),
        Attribute(float, "Müllerstraße", toml="muellerstrasse"),
        Attribute(dict, "Range", python="attack_range"),
        Attribute(list, "Test Attribute"),
        Attribute(list, "Z"),
        Attribute(list, "Z", aliases={"z1"}),
        Attribute(list, "Z", aliases={"z2"}),
        Attribute(list, "Z", toml="z1"),
        Attribute(list, "Z", toml="z2"),
        Attribute(list, "Z", python="z1"),
        Attribute(list, "Z", python="z2"),
    ]

    def test_eq(self):
        """Test Attribute == Attribute."""
        assert self.attributes[0] == self.attributes[0]
        assert not self.attributes[0] == self.attributes[-1]
        assert not self.attributes[-1] == self.attributes[0]
        assert self.attributes[-1] == self.attributes[-1]

    def test_ne(self):
        """Test Attribute != Attribute."""
        assert not self.attributes[0] != self.attributes[0]
        assert self.attributes[0] != self.attributes[-1]
        assert self.attributes[-1] != self.attributes[0]
        assert not self.attributes[-1] != self.attributes[-1]

    def test_lt(self):
        """Test Attribute < Attribute."""
        assert not self.attributes[0] < self.attributes[0]
        assert self.attributes[0] < self.attributes[-1]
        assert not self.attributes[-1] < self.attributes[0]
        assert not self.attributes[-1] < self.attributes[-1]

    def test_le(self):
        """Test Attribute <= Attribute."""
        assert self.attributes[0] <= self.attributes[0]
        assert self.attributes[0] <= self.attributes[-1]
        assert not self.attributes[-1] <= self.attributes[0]
        assert self.attributes[-1] <= self.attributes[-1]

    def test_gt(self):
        """Test Attribute > Attribute."""
        assert not self.attributes[0] > self.attributes[0]
        assert not self.attributes[0] > self.attributes[-1]
        assert self.attributes[-1] > self.attributes[0]
        assert not self.attributes[-1] > self.attributes[-1]

    def test_ge(self):
        """Test Attribute >= Attribute."""
        assert self.attributes[0] >= self.attributes[0]
        assert not self.attributes[0] >= self.attributes[-1]
        assert self.attributes[-1] >= self.attributes[0]
        assert self.attributes[-1] >= self.attributes[-1]

    def test_sort(self):
        """Test Attribute sorting."""
        assert self.attributes == sorted(self.attributes)
