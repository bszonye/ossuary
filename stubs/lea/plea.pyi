from typing import Any

from .lea import Lea

class Plea(Lea):
    def __getattr__(self, name: str) -> Any: ...  # incomplete
