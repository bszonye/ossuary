"""Unit tests for bones.warhammer.Profile class."""

import pytest

from bones.warhammer import Attribute, Profile


# Simple class hierarchy to test attribute inheritance.
class SubProfileA(Profile):  # noqa: D101
    CLASS_ATTRIBUTES = frozenset({Attribute("Test A", int)})


class SubProfileB(Profile):  # noqa: D101
    CLASS_ATTRIBUTES = frozenset({Attribute("Test B1"), Attribute("Test B2")})


class SubProfileC(Profile):  # noqa: D101
    pass  # No extra class attributes in this one.


class SubProfileABC(SubProfileA, SubProfileB, SubProfileC):  # noqa: D101
    CLASS_ATTRIBUTES = frozenset({Attribute("Test ABC", float)})


class TestProfileMeta:
    """Test the ProfileMeta metaclass."""

    def test_profile_attributes(self) -> None:
        """Test attribute initialization in the Profile base class."""
        assert type(Profile.ATTRIBUTES) == frozenset
        assert Profile.ATTRIBUTES == Profile.CLASS_ATTRIBUTES

    def test_metaclass(self) -> None:
        """Test metaclass initialization."""
        assert type(SubProfileABC.ATTRIBUTES) == frozenset
        assert SubProfileABC.ATTRIBUTES == (
            Profile.ATTRIBUTES
            | SubProfileA.CLASS_ATTRIBUTES
            | SubProfileB.CLASS_ATTRIBUTES
            # SubProfileB.CLASS_ATTRIBUTES -- no attributes
            | SubProfileABC.CLASS_ATTRIBUTES
        )


class TestProfileInit:
    """Test the Profile class constructor."""

    def test_simple(self) -> None:
        """Test with minimal arguments."""
        prof = Profile({})
        with pytest.raises(AttributeError):
            prof.name

    def test_name(self) -> None:
        """Test name parameter."""
        prof = Profile(dict(name="Test"))
        assert prof.name == "Test"


class TestProfileRepr:
    """Test the Profile.__repr__ special method."""

    @pytest.mark.xfail  # TODO
    def test_simple(self) -> None:
        """Test with minimal arguments."""
        prof = Profile({})
        out = "Profile({})"
        assert repr(prof) == out

    def test_name(self) -> None:
        """Test name parameter."""
        prof = Profile(dict(name="Test"))
        out = "Profile({'name': 'Test'})"
        assert repr(prof) == out
