"""Unit tests for bones.warhammer.Profile class."""

from bones.warhammer import Attribute, Profile


class SubProfileA(Profile):
    """Subclass of Profile for MRO testing."""

    CLASS_ATTRIBUTES = (Attribute(int, "Test A"),)


class SubProfileB(Profile):
    """Subclass of Profile for MRO testing."""

    CLASS_ATTRIBUTES = {
        Attribute(str, "Test B1"),
        Attribute(str, "Test B2"),
    }


class SubProfileC(Profile):
    """Subclass of Profile for MRO testing."""

    # No extra class attributes in this one.


class SubProfileABC(SubProfileA, SubProfileB, SubProfileC):
    """Subclass of Profile for MRO testing."""

    CLASS_ATTRIBUTES = frozenset({Attribute(int, "Test ABC")})


class TestProfileInit:
    """Test the Profile class constructor."""

    def test_simple(self):
        """Test with minimal arguments."""
        prof = Profile({})
        assert prof.name is None
        assert prof._attributes == frozenset()

    def test_custom_name(self):
        """Test name parameter."""
        prof = Profile({}, name="Test")
        assert prof._attributes == frozenset()
        assert prof.name == "Test"

    def test_custom_attributes(self):
        """Test attributes parameter."""
        attrs = frozenset(
            {
                Attribute(int, "Test 1"),
                Attribute(str, "Test 2"),
            }
        )
        prof = Profile({}, attributes=attrs)
        assert prof.name is None
        assert prof._attributes == attrs

    def test_metaclass(self):
        """Test metaclass initialization."""
        abc = frozenset(
            {
                Attribute(int, "Test A"),
                Attribute(str, "Test B1"),
                Attribute(str, "Test B2"),
                Attribute(int, "Test ABC"),
            }
        )
        assert type(SubProfileABC.ATTRIBUTES) == frozenset
        assert SubProfileABC.ATTRIBUTES == abc

    def test_inherit_default(self):
        """Test attribute inheritance."""
        prof = SubProfileABC({})
        assert prof.name is None
        assert prof._attributes == SubProfileABC.ATTRIBUTES

    def test_inherit_true(self):
        """Test attribute inheritance."""
        attrs = frozenset(
            {
                Attribute(int, "Test 1"),
                Attribute(str, "Test 2"),
            }
        )
        prof = SubProfileABC({}, attributes=attrs, inherit=True)
        assert prof.name is None
        assert prof._attributes == attrs | SubProfileABC.ATTRIBUTES

    def test_inherit_false(self):
        """Test attribute inheritance."""
        attrs = frozenset(
            {
                Attribute(int, "Test 1"),
                Attribute(str, "Test 2"),
            }
        )
        prof = SubProfileABC({}, attributes=attrs, inherit=False)
        assert prof.name is None
        assert prof._attributes == attrs


class TestProfileRepr:
    """Test the Profile.__repr__ special method."""

    def test_simple(self):
        """Test with minimal arguments."""
        prof = Profile({})
        out = "Profile({})"
        assert repr(prof) == out

    def test_custom_name(self):
        """Test name parameter."""
        prof = Profile({}, name="Test")
        out = "Profile({}, name='Test')"
        assert repr(prof) == out
