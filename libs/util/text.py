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
import spacy

def load_ner_model(language: Language):

    if language == Language.SPANISH:
        ner_model= "es_core_news_sm";
    else:
        ner_model = "en_core_web_sm"

    try:
        return spacy.load(ner_model)
    except OSError:
        print("Downloading en_core_web_sm model for spaCy...")
        spacy.cli.download(ner_model)
        return spacy.load(ner_model)
    
def get_stop_words(language: Language):
    if language == Language.SPANISH:
        return stopwords.words("spanish")
    elif language == Language.ENGLISH:
        return stopwords.words("english")
    