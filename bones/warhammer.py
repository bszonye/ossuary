"""bones.warhammer: dice analysis for Warhammer games."""

# from __future__ import annotations

__all__ = (
    "AttackCounter",
    "Attribute",
    "Datasheet",
    "Profile",
    "Unit",
    "UnitProfile",
    "Warscroll",
    "Weapon",
    "chain_rolls",
)

import builtins
import keyword
import sys
import unicodedata
from abc import ABCMeta
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, BinaryIO, ClassVar, Optional, Self, SupportsFloat, Union

import lea

# Conditionally import tomllib (Python 3.11+) or tomli.
# (mypy doesn't understand the try/except idiom)
if sys.version_info[:2] < (3, 11):
    import tomli as tomllib
else:
    import tomllib

# Type definitions.
Numeric = SupportsFloat  # reasonable approximation of "numeric"
AttributeFactory = Union[type[Any], Callable[..., Any]]
NameDict = dict[str, Any]
NameMapping = Mapping[str, Any]
NamePairs = Iterable[tuple[str, Any]]
Randomizable = Union[Numeric, lea.Lea]  # TODO: remove

lea.set_prob_type("r")


@dataclass(frozen=True)
class AttackCounter:
    """Results counter for each step of the attack sequence."""

    # TODO: addition operator?
    attacks: int
    wounds: int = 0
    mortals: int = 0
    kills: int = 0


@dataclass(frozen=True, order=True)
class Attribute:
    """Descriptor for profile attributes."""

    name: str  # human display name
    factory: AttributeFactory = field(default=str, compare=False)

    @staticmethod
    @cache
    def normalize(
        __s: str, /, *, connector: str = "_", python: Optional[bool] = None
    ) -> str:
        """Normalize string to an identifier or dictionary key.

        Normalization steps:
        * Apply the NFKC, casefold, and NFC transformations.
        * Replace non-identifier characters with the given connector.
        * Remove redundant connectors from the middle and ends.

        Python-specific rules:
        * Leading numerals get a guard prefix (e.g. A123_abc).
        * Builtin names and keywords get a guard suffix (e.g. range_).

        The extra Python rules apply if the connector is a legal
        xid_continue sequence and python is not False.  Raises
        ValueError if python is True but the connector is not legal
        in identifiers.
        """

        def is_xid_continue(__s: str, /) -> bool:
            """Check if a string is legal in identifiers."""
            return ("a" + __s).isidentifier()

        # Check Python identifier rules.
        if python is None:
            python = is_xid_continue(connector)
        elif python and not is_xid_continue(connector):
            raise ValueError(f"{connector!r} is not valid in identifiers")

        # First, normalize Unicode with NFKC -> casefold -> NFC.
        norm = unicodedata.normalize("NFKC", __s)
        norm = unicodedata.normalize("NFC", norm.casefold())

        # Blank all non-identifier characters.  Also blank any existing
        # Unicode connectors (Pc) in the input string, to help with
        # removing redundant connectors below.
        norm = "".join(
            " " if unicodedata.category(c) == "Pc" or not is_xid_continue(c) else c
            for c in norm
        )

        # Combine all normalized substrings with the given connector.
        # This uses split() to remove redundant connectors in the middle
        # and at both ends.
        norm = connector.join(norm.split())

        # Normalization is complete here for custom connectors.
        # Otherwise, continue to the Python-specific rules.
        if not python:
            return norm

        # Guard strings that start with an xid_continue character that
        # is not in xid_start.  Adds a capital letter prefix, as that
        # will not occur in a case-folded string.
        if not norm.isidentifier():
            norm = "A" + norm

        # Guard against collisions with builtins and keywords.  Adds an
        # underscore suffix as recommended by PEP 8.
        if norm in builtins.__dict__ or keyword.iskeyword(norm):
            norm += "_"

        return norm

    def python(self) -> str:
        """Return the attribute's normalized Python identifier."""
        return self.normalize(self.name)

    def toml(self) -> str:
        """Return the attribute's normalized TOML key."""
        return self.normalize(self.name, connector="-")

    def __repr__(self) -> str:
        """Dump attribute in repr() format."""
        args = [repr(self.name)]
        if self.factory != str:
            args.append(self.factory.__name__)
        # Join the result.
        return f"{type(self).__name__}({', '.join(args)})"


class ProfileMeta(ABCMeta):
    """Metaclass to manage profile attributes."""

    ATTRIBUTES: frozenset[Attribute]
    CLASS_ATTRIBUTES: Sequence[Attribute]

    def __new__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: NameDict,
    ) -> "ProfileMeta":
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, *args: Any, **kwargs: Any) -> None:  # noqa: B902
        """Initialize ATTRIBUTES for each Profile class."""
        super().__init__(*args, **kwargs)
        attrs: set[Attribute] = set()
        for c in cls.__mro__:
            attrs |= set(getattr(c, "CLASS_ATTRIBUTES", ()))
        cls.ATTRIBUTES = frozenset(attrs)


class Profile(NameMapping, metaclass=ProfileMeta):
    """Tabular data for units and weapons.

    Most Warhammer stats are **characteristics** organized into tables
    with a named **profile** in each row and the characteristic values
    in columns.  This base class represents basic profile data with
    subscript notation for accessing the characteristics by name.

    """

    # MERGED ATTRIBUTES:
    # The public class variable is specialized for each Profile subclass
    # to contain the union of all its class attributes with all of the
    # superclass attributes.  (Automatically set up by the metaclass.)
    ATTRIBUTES: ClassVar[frozenset[Attribute]]

    # Default attributes for this class.
    CLASS_ATTRIBUTES = frozenset({Attribute("Name")})
    name: str

    def __init__(self, __c: Union[NameMapping, NamePairs]):
        """Initialize characteristics."""
        # Convert input data to attributes.
        factories = {a.python(): a.factory for a in self.ATTRIBUTES}
        mapping = __c if isinstance(__c, Mapping) else dict(__c)
        for k, v in mapping.items():
            # TODO: Error handling / better error messages.
            attribute = Attribute.normalize(k)
            factory = factories[attribute]
            value = factory(v)
            setattr(self, attribute, value)

    @classmethod
    def load(cls, __fp: BinaryIO, /, *args: Any, **kwargs: Any) -> Self:
        """Construct a new Profile object from a TOML file."""
        # TODO: test this
        data = tomllib.load(__fp)
        return cls(data, *args, **kwargs)

    @classmethod
    def loads(cls, __s: str, /, *args: Any, **kwargs: Any) -> Self:
        """Construct a new Profile object from a TOML string."""
        # TODO: test this
        data = tomllib.loads(__s)
        return cls(data, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Look up an attribute by name, with normalization.

        Uses Attribute.normalize to convert a human-readable attribute
        name or TOML key to an equivalent Python attribute identifier.
        """
        norm = Attribute.normalize(name)
        try:
            return self.__dict__[norm]
        except KeyError:
            names = f"attribute {norm!r}"
            if norm != name:
                names += f"or {name!r}"
            error = f"{type(self).__name__!r} object has no {names}"
            if sys.version_info[:2] < (3, 10):
                raise AttributeError(error) from None
            raise AttributeError(error, name=norm, obj=self) from None

    def __getitem__(self, key: str) -> Any:
        """Subscript notation to simplify profile attribute access."""
        try:
            return getattr(self, key)
        except AttributeError as ex:
            raise KeyError(*ex.args) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of all profile attributes."""
        # TODO: test this
        return iter(a.python() for a in self.ATTRIBUTES)

    def __len__(self) -> int:
        """Report the number of profile attributes."""
        # TODO: test this
        return len(self.ATTRIBUTES)

    def __repr__(self) -> str:
        """Dump object data in repr() format."""
        args = [f"{dict(self.items())!r}"]
        return f"{type(self).__name__}({', '.join(args)})"


class Weapon(Profile):
    """Weapon profile."""


class UnitProfile(Profile):
    """Shared behavior for Warscroll and Datasheet objects."""


class Warscroll(UnitProfile):
    """Unit characteristics & abilities for Warhammer Age of Sigmar."""


class Datasheet(UnitProfile):
    """Unit characteristics & abilities for Warhammer 40,000."""


class Unit:
    """Fully specified army unit.

    Includes the unit profile (warscroll or datasheet), number of
    models, and all optional selections such as weapon options or
    upgrades.
    """


def chain_rolls(n: Randomizable, d: lea.Lea) -> lea.Lea:
    """Analyze one step in the attack sequence."""
    # n  number of incoming rolls (as pmf)
    # d  pmf for this roll
    nx: lea.Lea = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
