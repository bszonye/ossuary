"""Unit tests for the Profile class & subtypes."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

import io
from dataclasses import asdict, dataclass

import pytest

from bones.warhammer import Profile


# Simple class hierarchy to test attribute inheritance.
@dataclass
class SubProfileA(Profile):
    test_a: int = 0


@dataclass
class SubProfileB(Profile):
    test_b1: str = ""
    test_b2: str = ""


@dataclass
class SubProfileC(Profile):
    pass  # No extra class attributes in this one.


@dataclass
class SubProfileABC(SubProfileA, SubProfileB, SubProfileC):
    test_abc: float = 1.0


class TestProfileInheritance:
    def test_inherited_attributes(self) -> None:
        # Test attribute initialization in the Profile base class.
        subABC = SubProfileABC()
        assert {*subABC} == {"name", "test_a", "test_b1", "test_b2", "test_abc"}


class TestProfileInit:
    def test_defaults(self) -> None:
        prof = Profile()
        assert prof.name == type(prof).name

    def test_custom_name(self) -> None:
        # Test constructor with name parameter.
        prof = Profile("Test")
        assert prof.name == "Test"


class TestProfileLoad:
    def test_loadmap_trivial(self) -> None:
        # Test Profile.loadmap with an empty mapping.
        profiles = Profile.loadmap({})
        assert profiles == {}

    def test_loadmap_simple(self) -> None:
        # Test Profile.loadmap with just a name.
        profiles = Profile.loadmap({"Test Profile": {}})
        assert len(profiles) == 1
        prof = profiles["Test Profile"]
        assert prof.name == "Test Profile"

    def test_loadmap_attr(self) -> None:
        # Test Profile.loadmap with a profile set.
        profiles = Profile.loadmap({"Test Profile": {"name": "Test"}})
        assert len(profiles) == 1
        prof = profiles["Test Profile"]
        assert prof.name == "Test"

    def test_loadmap_multiple(self) -> None:
        # Test Profile.loadmap with multiple profiles
        profiles = Profile.loadmap(
            {
                "Test 1": {},
                "Test 2": {"name": "Test Two"},
                "Test 3": dict(name="Test Three"),
            }
        )
        assert len(profiles) == 3
        assert profiles["Test 1"].name == "Test 1"
        assert profiles["Test 2"].name == "Test Two"
        assert profiles["Test 3"].name == "Test Three"

    def test_loadmap_defaults(self) -> None:
        # Test Profile.loadmap with default values.
        profiles = SubProfileABC.loadmap(
            {
                "Test 1": {"test_a": 1},
                "Test 2": {"name": "Test Two", "test_b1": "baz"},
                "Test 3": dict(name="Test Three", test_abc=1.0),
            },
            name="NAME",
            test_a=42,
            test_b1="FOO",
            test_b2="BAR",
            test_abc=4.2,
        )
        assert len(profiles) == 3
        assert asdict(profiles["Test 1"]) == dict(
            name="NAME",
            test_a=1,
            test_b1="FOO",
            test_b2="BAR",
            test_abc=4.2,
        )
        assert asdict(profiles["Test 2"]) == dict(
            name="Test Two",
            test_a=42,
            test_b1="baz",
            test_b2="BAR",
            test_abc=4.2,
        )
        assert asdict(profiles["Test 3"]) == dict(
            name="Test Three",
            test_a=42,
            test_b1="FOO",
            test_b2="BAR",
            test_abc=1.0,
        )

    def test_loadmap_field_errors(self) -> None:
        # Unknown field name.
        with pytest.raises(ValueError) as ex:
            Profile.loadmap({"Test": {"name": "Test", "game": "Test"}})
        assert ex.value.args == ("unknown field 'game' in 'Test' Profile",)
        # Unknown non-normalized field name.
        with pytest.raises(ValueError) as ex:
            Profile.loadmap({"Test": {"NAME": "Test", "GAME": "Test"}})
        assert ex.value.args == (
            "unknown field 'GAME' (normalized 'game') in 'Test' Profile",
        )
        # Duplicate normalized field name.
        with pytest.raises(ValueError) as ex:
            Profile.loadmap({"Test": {"NAME": "Test", "name": "Test"}})
        assert ex.value.args == ("duplicate field 'name' in 'Test' Profile",)
        # Duplicate non-normalized field name.
        with pytest.raises(ValueError) as ex:
            Profile.loadmap({"Test": {"Name": "Test", "NAME": "Test"}})
        assert ex.value.args == (
            "duplicate field 'NAME' (normalized 'name') in 'Test' Profile",
        )

    TOML = """
    [Test]
    name = "Test 1"

    ["Test 2"]

    [test3]
    name = 3.0
    """

    def test_loads(self) -> None:
        profiles = Profile.loads(self.TOML)
        assert len(profiles) == 3
        assert profiles["Test"].name == "Test 1"
        assert profiles["Test 2"].name == "Test 2"
        assert profiles["test3"].name == "3.0"  # convert from float

    def test_loadf(self) -> None:
        bf = io.BytesIO(bytes(self.TOML, "utf-8"))
        profiles = Profile.loadf(bf)
        assert len(profiles) == 3
        assert profiles["Test"].name == "Test 1"
        assert profiles["Test 2"].name == "Test 2"
        assert profiles["test3"].name == "3.0"  # convert from float


class TestProfileFields:
    def test_simple(self) -> None:
        # Test with the base Profile class.
        prof = Profile()
        fmap = prof.fields()
        # Instance and class should produce the same fields.
        assert fmap == Profile.fields()  # by value
        assert fmap is Profile.fields()  # by cache
        # Check field contents.
        assert len(fmap) == 1
        fname = fmap["name"]
        assert fname.name == "name"
        assert fname.type == str

    def test_inherited_fields(self) -> None:
        # Test attribute initialization in the Profile base class.
        subABC = SubProfileABC()
        fmap = subABC.fields()
        # Instance and class should produce the same fields.
        assert fmap == SubProfileABC.fields()  # by value
        assert fmap is SubProfileABC.fields()  # by cache
        # Check field contents.
        assert len(fmap) == 5
        assert {*fmap} == {"name", "test_a", "test_b1", "test_b2", "test_abc"}


class TestProfileNormalize:
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
        python, null, nopython, toml, colon = self.normalize_table[name]
        assert python == Profile.normalize(name)
        assert python == Profile.normalize(name, connector="_")
        assert null == Profile.normalize(name, connector="")
        assert nopython == Profile.normalize(name, python=False)
        assert toml == Profile.normalize(name, connector="-")
        assert colon == Profile.normalize(name, connector=":")

    def test_normalize_error(self) -> None:
        with pytest.raises(ValueError):
            Profile.normalize("", connector=":", python=True)


class TestProfileMapping:
    def test_sequence(self) -> None:
        # Test Profile sequence operations: len, iter.
        subABC = SubProfileABC()
        assert len(subABC) == 5
        assert len(subABC) == len(tuple(subABC))

    def test_mapping(self) -> None:
        subABC = SubProfileABC(name="Test")
        assert subABC["name"] == "Test"  # exact name lookup
        assert subABC["NAME"] == "Test"  # normalized name lookup
        assert subABC["__name__"] == "Test"  # normalized name lookup

    def test_mapping_error(self) -> None:
        subABC = SubProfileABC(name="Test")
        with pytest.raises(KeyError):
            subABC["game"]  # exact name lookup
        with pytest.raises(KeyError):
            subABC["GAME"]  # normalized name lookup

    def test_attribute_error(self) -> None:
        subABC = SubProfileABC(name="Test")
        with pytest.raises(AttributeError):
            subABC.game  # exact name lookup
        with pytest.raises(AttributeError):
            subABC.GAME  # normalized name lookup
