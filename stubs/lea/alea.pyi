from typing import Any, Iterable, Union

from .lea import Lea

_mapping = Union[dict[Any, Any], Iterable[tuple[Any, Any]]]

class Alea(Lea):
    @staticmethod
    def pmf(arg: _mapping, prob_type: str = ..., **kwargs: Any) -> Alea: ...
    @staticmethod
    def set_prob_type(prob_type: str) -> None: ...
    @staticmethod
    def vals(*values: Any, **kwargs: Any) -> Alea: ...
    def __getattr__(self, name: str) -> Any: ...  # incomplete
