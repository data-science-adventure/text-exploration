
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber  # For more robust text extraction, especially with layouts
from pathlib import Path
from model.context import Language, FileType, Context
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from model.context import UnitOfAnalysis
from model.document import Document

def read_documents_as_stream(directory: Path):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Read the combined cleaned text for further analysis
            file_to_process = directory / Path(filename)
            with open(file_to_process, "r", encoding="utf-8") as f:
                yield f.read()


def read_documents(context: Context):
    documents: list[Document] = []
    for filename in os.listdir(context.extraction_context.directory):
        if filename.endswith(".txt"):
            # Read the combined cleaned text for further analysis
            file_to_process = context.extraction_context.directory / Path(filename)
            with open(file_to_process, "r", encoding="utf-8") as f:
                document = Document()
                document.name = filename
                document.unit_of_analysis = UnitOfAnalysis.WORDS
                text_content = f.read()
                document.words = word_tokenize(text_content)
                document.sentences = sent_tokenize(text_content)
                documents.append(document)
    return documents


def read_all_input_documents_as_one(directory: Path):
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_to_process = directory / Path(filename)
            with open(file_to_process, "r", encoding="utf-8") as f:
                text_content = f.read()
                all_documents.append(text_content)
    return "/n".join(all_documents)