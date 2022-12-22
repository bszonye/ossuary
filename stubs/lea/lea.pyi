import fractions
from typing import Any, Callable

from .alea import Alea
from .tlea import Tlea

class Lea:
    class Error(Exception): ...

    def switch(
        self, lea_dict: dict[Any, Any], default_lea: Lea = ..., prior_lea: Lea = ...
    ) -> Tlea: ...
    def __getattr__(self, name: str) -> Any: ...  # incomplete

# import modules with Lea subclasses
# these must be placed here to avoid cycles (these import lea module)
from .alea import Alea
from .blea import Blea
from .clea import Clea
from .dlea import Dlea
from .evidence_ctx import EvidenceCtx
from .flea import Flea
from .flea1 import Flea1
from .flea2 import Flea2
from .flea2a import Flea2a
from .glea import Glea
from .ilea import Ilea
from .olea import Olea
from .plea import Plea
from .rlea import Rlea
from .slea import Slea
from .tlea import Tlea

_lea_leaf_classes = (Alea, Olea, Plea)

def P(lea1: Lea) -> float | fractions.Fraction: ...
def Pf(lea1: Lea) -> float: ...
def __getattr__(name: str) -> Any: ...  # incomplete
