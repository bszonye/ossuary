from typing import Any

from .lea import Lea

class Olea(Lea):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
