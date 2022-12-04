"""bones.warhammer: dice analysis for Warhammer games."""

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

import dataclasses
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING
from typing import Any, IO, Optional, SupportsFloat, Type, TypeVar, Union

import lea

# Conditionally import tomllib (Python 3.11+) or tomli.
# (mypy doesn't understand the try/except idiom)
if sys.version_info[:2] < (3, 11):
    import tomli as tomllib
else:
    import tomllib

# Type definitions.
# TODO: remove unused types
AttributeFactory = Callable[[Any], Any]
Numeric = SupportsFloat  # reasonable approximation of "numeric"
Distance = Numeric  # e.g. for move & range characteristics
ProfileType = TypeVar("ProfileType", bound="Profile")
Randomizable = Union[Numeric, lea.Lea]
NameDict = dict[str, Any]
NameMapping = Mapping[str, Any]

lea.set_prob_type("r")


@dataclasses.dataclass(frozen=True)
class AttackCounter:
    """Results counter for each step of the attack sequence."""

    # TODO: addition operator?
    attacks: int
    wounds: int = 0
    mortals: int = 0
    kills: int = 0


class Attribute:
    """Descriptor for profile attributes."""

    factory: AttributeFactory  # type constructor
    name: str  # human display name
    python: str  # Python attribute name
    toml: str  # preferred TOML key name
    aliases: frozenset[str]  # list of acceptable aliases

    def __init__(
        self,
        factory: AttributeFactory,
        name: str,
        python: Optional[str] = None,
        toml: Optional[str] = None,
        aliases: Iterable[str] = (),
    ) -> None:
        """Construct an attribute descriptor.

        Each attribute requires a one-argument callable (such as a type
        constructor) to convert and validate input data.  They may also
        need specialized names for Python code and TOML keys, or
        additional aliases for ease of use.
        """
        self.factory = factory
        self.name = name
        self.python = python if python else self.codify(name)
        self.toml = toml if toml else self.tomlize(name)
        self.aliases = self.autoalias(self.name, self.python, self.toml, *aliases)

    @classmethod
    def codify(cls, __s: str, /) -> str:
        """Convert string to a Python identifier.

        * Fold case.
        * Consolidate non-alphanumerics to single underscores.
        * Remove leading underscores and guard leading digits.
        """
        code = __s.casefold()
        code = re.sub(r"[\W_]+", "_", code).lstrip("_")
        code = re.sub(r"^(?=\d)", "n", code)
        return code

    @classmethod
    def tomlize(cls, __s: str, /) -> str:
        """Convert string to a TOML key.

        * Fold case.
        * Consolidate non-alphanumerics to single hyphens.
        * Convert non-ASCII characters to underscores.
        """
        toml = __s.casefold()
        toml = re.sub(r"[\W_]+", "-", toml)
        toml = re.sub(r"[^-0-9A-Za-z]", "_", toml)
        return toml

    @classmethod
    def autoalias(cls, *args: str) -> frozenset[str]:
        """Create lookup aliases from a sequence of names.

        This find the closure of casefold, codify, and tomlize.  The
        alias lookup function uses case-insensitive matching with
        casefold, and the other two convert aliases to legal Python
        identifiers and TOML keys.  The algorithm finds the closure of
        all transformations for reliability and to make the operation
        more reversible for the __repr__ method.
        """
        aliases = set(args)
        closure: set[str] = set()
        while aliases != closure:  # Stop when the closure is stable.
            aliases |= closure
            for alias in aliases:
                closure |= {
                    alias,
                    alias.casefold(),
                    cls.codify(alias),
                    cls.tomlize(alias),
                }
        return frozenset(closure)

    def __repr__(self) -> str:
        """Dump attribute in repr() format."""
        args = [
            self.factory.__name__,
            repr(self.name),
        ]
        # Add non-default arguments.
        if self.python != self.codify(self.name):
            args.append(f"python={self.python!r}")
        if self.toml != self.tomlize(self.name):
            args.append(f"toml={self.toml!r}")
        aliases = self.aliases - self.autoalias(self.name, self.python, self.toml)
        if aliases:
            # Remove redundant aliases.
            minset = set(aliases)
            for alias in aliases:
                extra = self.autoalias(alias) - {alias}
                minset -= extra
            args.append(f"aliases={tuple(sorted(minset))!r}")
        # Join the result.
        return f"{type(self).__name__}({', '.join(args)})"


class Profile(NameMapping):
    """Tabular data for units and weapons.

    Most Warhammer stats are **characteristics** organized into tables
    with a named **profile** in each row and the characteristic values
    in columns.  This base class represents basic profile data with
    subscript notation for accessing the characteristics by name.

    """

    name: Optional[str]
    _attributes: Sequence[Attribute] = ()
    _aliases: dict[str, str] = {}

    def __init__(
        self,
        __c: Union[NameDict, Iterable[Any]],
        /,
        *,
        name: Optional[str] = None,
    ):
        """Initialize characteristics."""
        # Convert input data to a characteristic dictionary.
        # TODO: initialize ATTR_MAP
        # TODO: initialize attributes
        self.name = name

    @classmethod
    def load(
        cls: Type[ProfileType], __fp: IO[bytes], /, *args: Any, **kwargs: Any
    ) -> ProfileType:
        """Construct a new Profile object from a TOML file."""
        # TODO: test this
        data = tomllib.load(__fp)
        return cls(data, *args, **kwargs)

    @classmethod
    def loads(
        cls: Type[ProfileType], __s: str, /, *args: Any, **kwargs: Any
    ) -> ProfileType:
        """Construct a new Profile object from a TOML string."""
        # TODO: test this
        data = tomllib.loads(__s)
        return cls(data, *args, **kwargs)

    def alias(self, alias: str) -> Optional[str]:
        """Get an attribute name from one of its aliases."""
        # TODO: initialize self._aliases if it isn't already done
        return self._aliases.get(alias.casefold())

    def attribute(
        self,
        alias: str,
        default: Any = MISSING,
    ) -> Any:
        """Find an attribute from one of its aliases."""
        # Look up the attribute from the alias.
        attribute = self.alias(alias)
        if attribute is None:
            if default is MISSING:
                error = "'{}' object has no attribute or alias '{}'".format(
                    type(self).__name__, alias
                )
                if sys.version_info[:2] < (3, 10):
                    raise AttributeError(error)
                raise AttributeError(error, name=alias, obj=self)
            return default
        # Return the attribute value.
        value = getattr(self, attribute, default)
        if value is MISSING:
            name = "attribute '{}'".format(attribute)
            if attribute != alias:
                name += " (alias '{}')".format(alias)
            error = "'{}' object has no {}".format(type(self).__name__, name)
            if sys.version_info[:2] < (3, 10):
                raise AttributeError(error)
            raise AttributeError(error, name=alias, obj=self)
        return value

    def __getitem__(self, key: str) -> Any:
        """Subscript notation to simplify profile attribute access."""
        try:
            return self.attribute(key)
        except AttributeError as ex:
            raise KeyError(*ex.args) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of all profile attributes."""
        # TODO: test this
        return iter(a.python for a in self._attributes)

    def __len__(self) -> int:
        """Report the number of profile attributes."""
        # TODO: test this
        return len(self._attributes)

    def __repr__(self) -> str:
        """Dump object data in repr() format."""
        # TODO: test this
        return "{}({}, name={})".format(
            type(self).__name__,
            repr(dict(self.items())),
            repr(self.name),
        )


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
    nx = n if isinstance(n, lea.Lea) else lea.vals(n)
    switch = {nv: d.times(nv) if nv else 0 for nv in nx.support}
    return nx.switch(switch)
