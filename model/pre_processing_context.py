from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class PreProcessingContext:
    directory: Optional[Path] = None