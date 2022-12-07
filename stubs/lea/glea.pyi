from typing import Any

from .lea import Lea

class Glea(Lea):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
