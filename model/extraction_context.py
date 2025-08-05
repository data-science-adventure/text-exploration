from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ExtractionContext:
    directory: Optional[Path] = None
    suffix: Optional[str] = None
