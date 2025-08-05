from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class FeatureExtractionContext:
    directory: Optional[Path] = None