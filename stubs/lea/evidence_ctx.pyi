from typing import Any

class EvidenceCtx:
    def __getattr__(self, name: str) -> Any: ...  # incomplete
