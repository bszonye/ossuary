"""bones.draw: bag & deck draw analysis."""

__author__ = "Bradd Szonye <bszonye@gmail.com>"

from typing import Generic, TypeVar

ET_co = TypeVar("ET_co", covariant=True)  # Covariant event type.


class Bag(Generic[ET_co]):
    """Model for drawing items randomly from a bag."""


class Deck(Generic[ET_co]):
    """Model for drawing cards randomly from a deck."""
