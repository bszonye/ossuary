from typing import Any, Iterable, Sequence

from .lea import Lea

_mapping = dict[Any, Any] | Iterable[tuple[Any, Any]]

class Alea(Lea):
    @staticmethod
    def set_prob_type(prob_type: str) -> None: ...
    def __init__(
        self,
        vs: Sequence[Any],
        ps: Sequence[Any],
        normalization: bool = ...,
        prob_type: Any = ...,
    ) -> None: ...
    @staticmethod
    def pmf(arg: _mapping, prob_type: str = ..., **kwargs: Any) -> Alea: ...
    @staticmethod
    def vals(*values: Any, **kwargs: Any) -> Alea: ...
    def __getattr__(self, name: str) -> Any: ...  # incomplete
