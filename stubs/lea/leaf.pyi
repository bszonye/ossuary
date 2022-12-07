from typing import Any

from .alea import Alea
from .lea import Lea

def die(nb_faces: int = ..., prob_type: str = ...) -> Alea: ...
def dice(nb_dice: int, nb_faces: int = ..., prob_type: str = ...) -> Lea: ...
def dice_seq(
    nb_dice: int, nb_faces: int = ..., sorted: bool = ..., prob_type: str = ...
) -> Alea: ...

D6: Alea
flip: Alea
card_suite: Alea
card_rank: Alea
card: Alea

def __getattr__(name: str) -> Any: ...  # incomplete
