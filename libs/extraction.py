import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber  # For more robust text extraction, especially with layouts
from pathlib import Path
from model.context import Language, FileType, Context
from nltk.corpus import stopwords
from model.document import Document
from libs.util.text import get_stop_words

def extract_text(path: Path, file_type: FileType):
    if file_type == FileType.PDF:
        return extract_text_from_pdf(path)
    else:
        return "Not supported"


def extract_text_from_pdf(pdf_path: Path):
    """
    Extracts text from a single PDF file using pdfplumber for better accuracy.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def remove_stop_words(documents: list[Document], language: Language):
    for document in documents:
        stop_words = set(get_stop_words(language))
        document.filtered_words = [
            word for word in document.words if word.isalnum() and word not in stop_words
        ]

def clean_text(text: str, language: Language):
    """
    Cleans the extracted text by removing extra whitespaces, special characters,
    and converting to lowercase.
    """
    if not text:
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Replace multiple whitespaces with a single space
    if language == Language.SPANISH:
        text = re.sub(r'[^a-z0-9\s.,;\'"-áéíóúüñ]', "", text)
    else:
        text = re.sub(
            r'[^a-z0-9\s.,;\'"-]', "", text
        )  # Keep alphanumeric, spaces, and common punctuation
    text = text.strip()  # Remove leading/trailing whitespaces
    return text

def extract_pdfs_pipeline(context: Context):
    """
    Pipeline to read multiple PDFs, extract, clean, and store text.
    Combines all cleaned text into a single file.
    """
    for filename in os.listdir(context.input_data_directory):
        if filename.endswith(".pdf"):

            pdf_path = context.input_data_directory / Path(filename)
            print(f"Processing {filename}...")
            extracted_text = extract_text(pdf_path, context.file_type)
            cleaned_txt = clean_text(extracted_text, context.language)

            # Store individual cleaned text files (optional, but good for debugging)
            output_txt_filename = pdf_path.stem + context.extraction_context.suffix + ".txt"
            output_txt_path = context.extraction_context.directory / Path(output_txt_filename)
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_txt)
            print(f"Cleaned text saved to {output_txt_path}")
