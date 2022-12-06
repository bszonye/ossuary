"""bones.warhammer: dice analysis for Warhammer games."""

# TODO: Here for reference. Remove if possible.
# from __future__ import annotations

__all__ = (
    "AttackCounter",
    "Datasheet",
    "Profile",
    "Unit",
    "UnitProfile",
    "Warscroll",
    "Weapon",
)

import builtins
import functools
import keyword
import tomllib
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, Field, fields
from typing import Any, BinaryIO, Optional, Self

import lea

# Type definitions.
NameMapping = Mapping[str, Any]

lea.set_prob_type("r")


@dataclass(frozen=True)
class AttackCounter:
    """Results counter for each step of the attack sequence."""

    attacks: int
    wounds: int = 0
    mortals: int = 0
    kills: int = 0


@dataclass
class Profile(NameMapping):
    """Tabular data for units and weapons.

    Most Warhammer stats are **characteristics** organized into tables
    with a named **profile** in each row and the characteristic values
    in columns.  This base class represents basic profile data with
    subscript notation for accessing the characteristics by name.

    """

    name: str = "Untitled"

    @classmethod
    def loadmap(cls, __map: NameMapping, /) -> Mapping[str, Self]:
        """Construct a new Profile object from a mapping."""
        profiles: dict[str, Self] = {}
        cname = cls.__name__
        fmap = cls.fields()

        # Each value in __map should also be a NameMapping.
        pname: str
        pdata: NameMapping
        for pname, pdata in __map.items():
            pmap: dict[str, Any] = {}
            fname: str

            # Collect the profile fields.
            for fname, fdata in pdata.items():
                attr = cls.normalize(fname)
                if attr not in fmap or attr in pmap:
                    # Bad field name.  Either there's no matching field,
                    # or two names normalized onto the same attribute.
                    problem = "duplicate" if attr in fmap else "unknown"
                    falias = repr(fname)
                    if attr != fname:
                        falias += f" (normalized {attr!r})"
                    error = f"{problem} field {falias} in {pname!r} {cname}"
                    raise ValueError(error)
                # Convert the field data to the field type.
                ftype = fmap[attr].type
                pmap[attr] = ftype(fdata)

            # Set the profile name if there wasn't a name field given.
            if "name" not in pmap:
                pmap["name"] = pname
            # Add the profile to the output.
            profile = cls(**pmap)
            profiles[pname] = profile

        return profiles

    @classmethod
    def loadf(
        cls, __fp: BinaryIO, /, *args: Any, **kwargs: Any
    ) -> Mapping[str, Self]:
        """Construct a new Profile object from a TOML file."""
        # TODO: test this
        data = tomllib.load(__fp)
        return cls.loadmap(data)

    @classmethod
    def loads(cls, __s: str, /, *args: Any, **kwargs: Any) -> Mapping[str, Self]:
        """Construct a new Profile object from a TOML string."""
        # TODO: test this
        data = tomllib.loads(__s)
        return cls.loadmap(data)

    @staticmethod
    @functools.cache
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
        if (
            norm in builtins.__dict__
            or keyword.iskeyword(norm)
            or keyword.issoftkeyword(norm)
        ):
            norm += "_"

        return norm

    @classmethod
    @functools.cache
    def fields(cls) -> dict[str, Field[Any]]:
        """Index the class fields by name."""
        return {f.name: f for f in fields(cls)}

    def __getattr__(self, __name: str) -> Any:
        """Look up an attribute by name, with normalization.

        Uses Attribute.normalize to convert a human-readable attribute
        name or TOML key to an equivalent Python attribute identifier.
        """
        name = self.normalize(__name)
        try:
            return self.__dict__[name]
        except KeyError:
            names = f"attribute {name!r}"
            if name != __name:
                names += f"or {__name!r}"
            error = f"{type(self).__name__!r} object has no {names}"
            raise AttributeError(error, name=name, obj=self) from None

    def __getitem__(self, __key: str) -> Any:
        """Subscript notation to simplify profile attribute access."""
        try:
            return getattr(self, __key)
        except AttributeError as ex:
            raise KeyError(*ex.args) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of all profile attributes."""
        return iter(f.name for f in fields(self))

    def __len__(self) -> int:
        """Report the number of profile attributes."""
        return len(fields(self))


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
