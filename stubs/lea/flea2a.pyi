from typing import Any

from .flea2 import Flea2

class Flea2a(Flea2):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
