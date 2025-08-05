from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from model.context import UnitOfAnalysis

@dataclass
class Document:
    name: Optional[str] = None
    unit_of_analysis: Optional[UnitOfAnalysis] = None
    words: Optional[list[str]] = None
    sentences: Optional[list[str]] = None
    filtered_words: Optional[list[str]] = None
