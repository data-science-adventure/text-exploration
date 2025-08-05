import os
from pathlib import Path
from dataclasses import dataclass, field
from model.file import File
from model.extraction_context import ExtractionContext
from model.pre_processing_context import PreProcessingContext
from model.feature_extraction_context import FeatureExtractionContext
from typing import Optional
from enum import Enum


# Define the Enum class
class Language(Enum):
    SPANISH = 1
    ENGLISH = 2


class FileType(Enum):
    PDF = 1
    DOC = 2


class UnitOfAnalysis(Enum):
    WORDS = 1
    PHRASES = 2
    SENTENCES = 3
    PARAGRAPHS = 4


@dataclass
class Context:
    language: Optional[Language] = None
    file_type: Optional[FileType] = None
    unit_of_analysis: Optional[UnitOfAnalysis] = None
    data_directory: Optional[Path] = None
    input_data_directory: Optional[Path] = None
    file_to_process: Optional[Path] = None
    extraction_context: Optional[ExtractionContext] = None
    pre_processing_context: Optional[PreProcessingContext] = None
    feature_extraction_context: Optional[FeatureExtractionContext] = None
    words_to_track: Optional[list[str]] = None

    def print_properties(self):
        """
        Prints every instance property of the Context class,
        and recursively prints properties for nested custom objects.
        """
        print("--- Context Properties ---")
        for key, value in self.__dict__.items():
            # Check if the value is a custom class instance
            # and print its properties in a nested format.
            if isinstance(
                value,
                (
                    File,
                    ExtractionContext,
                    PreProcessingContext,
                    FeatureExtractionContext,
                ),
            ):
                print(f"{key}:")
                # Iterate through the nested object's properties
                for sub_key, sub_value in value.__dict__.items():
                    print(f"  - {sub_key}: {sub_value}")
            else:
                # If it's a simple type, just print it directly.
                print(f"{key}: {value}")
        print("--------------------------")

    def __repr__(self) -> str:
        """
        A custom __repr__ method for a more readable, multi-line output.
        """
        # Create a list of lines for each attribute
        lines = [f"{key}: {value}" for key, value in self.__dict__.items()]

        # Join the lines with a newline and indentation
        return "Context(\n" + "  " + "\n  ".join(lines) + "\n)"


class ContextBuilder:
    def __init__(self):
        # Default values
        self.with_default()

    def with_default(self):
        data_directory = Path.cwd() / "data"

        self.context = Context(
            language=Language.SPANISH,
            file_type=FileType.PDF,
            unit_of_analysis=UnitOfAnalysis.WORDS,
            data_directory=data_directory,
            input_data_directory=data_directory / "input",
            extraction_context=ExtractionContext(
                data_directory / "extraction", suffix="-raw"
            ),
            pre_processing_context=PreProcessingContext(
                data_directory / "pre_processing"
            ),
            feature_extraction_context=FeatureExtractionContext(
                data_directory / "feature_extraction"
            ),
            words_to_track=[],
        )
        return self

    def with_data_directory_name(self, data_directory_name):
        data_directory = Path.cwd() / data_directory_name
        self.context.data_directory = data_directory
        self.context.extraction_context.directory = (
            data_directory / self.context.extraction_context.directory.name
        )
        self.context.pre_processing_context.directory = (
            data_directory / self.context.pre_processing_context.directory.name
        )
        self.context.feature_extraction_context.directory = (
            data_directory / self.context.feature_extraction_context.directory.name
        )
        self.context.words_to_track = []
        return self

    def with_extraction_directory_name(self, directory_name):
        self.context.extraction_context.directory = (
            self.context.data_directory / directory_name
        )
        return self

    def with_pre_processing_directory_name(self, directory_name):
        self.context.pre_processing_context.directory = (
            self.context.data_directory / directory_name
        )
        return self

    def with_feature_extraction_directory_name(self, directory_name):
        self.context.feature_extraction_context.directory = (
            self.context.data_directory / directory_name
        )
        return self

    def with_input_data_directory_name(self, directory_name):
        self.context.input_data_directory = (
            self.context.input_data_directory / directory_name
        )
        return self

    def with_language(self, language: Language):
        self.context.language = language
        return self

    def with_file_type(self, file_type: FileType):
        self.context.file_type = file_type
        return self

    def with_unit_of_analysis(self, unit_of_analysis: UnitOfAnalysis):
        self.context.unit_of_analysis = unit_of_analysis
        return self

    def with_words_to_track(self, words: list[str]):
        self.context.words_to_track = words
        return self

    def mkdir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

    def build(self):
        self.mkdir(self.context.data_directory)
        self.mkdir(self.context.input_data_directory)
        self.mkdir(self.context.extraction_context.directory)
        self.mkdir(self.context.feature_extraction_context.directory)
        self.mkdir(self.context.pre_processing_context.directory)
        return self.context
