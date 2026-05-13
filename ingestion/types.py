from dataclasses import dataclass
from typing import Dict


@dataclass
class PageUnit:
    id: str
    content: str
    unit_type: str
    index: int
    source: str

    def metadata(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "unit_type": self.unit_type,
            "index": self.index,
        }
