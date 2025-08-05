import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class File:
    directory: Optional[str] = None
    name: Optional[str] = None
    simple_name: Optional[str] = None

    def get_canonical_name(self):
        return os.path.join(self.directory, self.name)

    def __repr__(self) -> str:
        return f"File(\n  name='{self.name}',\n  directory='{self.directory}',\n  simple_name='{self.simple_name}',\n)"


class FileBuilder:
    def __init__(self):
        self.file = File()

    def with_name(self, name: str):
        self.file.name = name
        return self  # Return self to allow method chaining

    def with_directory(self, directory: str):
        self.file.directory = directory
        return self

    def build(self) -> File:
        """
        Constructs and returns the File object.
        """
        # Ensure that the minimum required parameters have been set
        if self.file.name is None or self.file.directory is None:
            raise ValueError(
                "Both name and directory must be set to build the File object."
            )

        # Derive the other properties from the name and directory
        self.file.simple_name = Path(self.file.name).stem

        # Instantiate the File dataclass and return it
        return self.file
