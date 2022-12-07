from typing import Any

from .lea import Lea

class Flea2(Lea):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
