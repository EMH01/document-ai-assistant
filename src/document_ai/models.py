from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Chunk:
    text: str
    page: int
    kind: str = "text"
    source: str = "document"
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def citation(self) -> str:
        label = "image" if self.kind == "image" else "page"
        return f"{self.source} · {label} {self.page}"
