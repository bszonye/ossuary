"""Unit tests for bones.warhammer.Profile class."""

from dataclasses import dataclass

import pytest

from bones.warhammer import Profile


# Simple class hierarchy to test attribute inheritance.
@dataclass
class SubProfileA(Profile):  # noqa: D101
    test_a: int = 0


@dataclass
class SubProfileB(Profile):  # noqa: D101
    test_b1: str = ""
    test_b2: str = ""


@dataclass
class SubProfileC(Profile):  # noqa: D101
    pass  # No extra class attributes in this one.


@dataclass
class SubProfileABC(SubProfileA, SubProfileB, SubProfileC):  # noqa: D101
    test_abc: float = 1.0


class TestProfileInheritance:
    """Test Profile attribute inheritance."""

    def test_inherited_attributes(self) -> None:
        """Test attribute initialization in the Profile base class."""
        prof = Profile()
        subA = SubProfileA()
        subB = SubProfileB()
        subC = SubProfileC()
        subABC = SubProfileABC()
        assert len(prof) == 1
        assert len(subA) == 2
        assert len(subB) == 3
        assert len(subC) == 1
        assert len(subABC) == 5
        assert {*subABC} == {"name", "test_a", "test_b1", "test_b2", "test_abc"}


class TestProfileInit:
    """Test the Profile class constructor."""

    def test_simple(self) -> None:
        """Test with minimal arguments."""
        prof = Profile("Test")
        assert prof.name == "Test"


class TestProfileNormalize:
    """Test the Profile.normalize method."""

    # input: (python, null, nopython, toml, colon)
    normalize_table = {
        # No transformation required.
        "abc": ("abc", "abc", "abc", "abc", "abc"),
        # Case folding.
        "ABC": ("abc", "abc", "abc", "abc", "abc"),
        "Große": ("grosse", "grosse", "grosse", "grosse", "grosse"),
        # Connect words.
        "a b c d": ("a_b_c_d", "abcd", "a_b_c_d", "a-b-c-d", "a:b:c:d"),
        "a b c": ("a_b_c", "abc", "a_b_c", "a-b-c", "a:b:c"),
        "a b": ("a_b", "ab", "a_b", "a-b", "a:b"),
        # Replace non-identifier characters with connectors.
        "abc-123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "abc/123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "abc:123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        # Replace existing connectors (Unicode category Pc).
        "abc_123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "abc‿123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "abc⁀123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "abc⁔123": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        # Remove redundant connectors.
        "  abc  123  ": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "+-abc-+-123-+": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        "==abc===123==": ("abc_123", "abc123", "abc_123", "abc-123", "abc:123"),
        # Python rules: first character is not xid_start.
        "123 abc": ("A123_abc", "A123abc", "123_abc", "123-abc", "123:abc"),
        "[123-abc]": ("A123_abc", "A123abc", "123_abc", "123-abc", "123:abc"),
        "__123_abc__": ("A123_abc", "A123abc", "123_abc", "123-abc", "123:abc"),
        # Python rules: name is a builtin or keyword.
        "break": ("break_", "break_", "break", "break", "break"),
        "match": ("match_", "match_", "match", "match", "match"),
        "range": ("range_", "range_", "range", "range", "range"),
        # Python rules: these are OK because of case folding!
        "False": ("false", "false", "false", "false", "false"),
        "None": ("none", "none", "none", "none", "none"),
        "True": ("true", "true", "true", "true", "true"),
        # Corner cases.
        "": ("A", "A", "", "", ""),
        "--": ("A", "A", "", "", ""),
        "_": ("A", "A", "", "", ""),
    }

    @pytest.mark.parametrize("name", normalize_table)
    def test_normalize(self, name: str) -> None:
        """Test Profile.normalize method with minimal arguments."""
        python, null, nopython, toml, colon = self.normalize_table[name]
        assert python == Profile.normalize(name)
        assert python == Profile.normalize(name, connector="_")
        assert null == Profile.normalize(name, connector="")
        assert nopython == Profile.normalize(name, python=False)
        assert toml == Profile.normalize(name, connector="-")
        assert colon == Profile.normalize(name, connector=":")

    def test_normalize_error(self) -> None:
        """Test Profile.normalize with incompatible parameters."""
        with pytest.raises(ValueError):
            Profile.normalize("", connector=":", python=True)
