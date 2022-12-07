from typing import Any

from .lea import Lea

class Flea(Lea):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
