from model.context import Language
from model.context import KeyphraseAlgorithm
import libs.extraction as extraction

from sklearn.feature_extraction.text import TfidfVectorizer
import textacy
import textacy.extract


def extract_keyphrases(
    text: str, language: Language, algorithm: KeyphraseAlgorithm, limit=50
):

    if algorithm == KeyphraseAlgorithm.TF_IDF:
        return with_tf_idf(text, language, limit)
    elif (
        algorithm == KeyphraseAlgorithm.TEXT_RANK
        or algorithm == KeyphraseAlgorithm.SG_RANK
    ):
        return with_text_or_sg_rank(text, language, algorithm, limit)


def with_tf_idf(text: str, language: Language, limit=50):
    vectorizer = TfidfVectorizer(
        stop_words=extraction.get_stop_words(language), max_features=limit
    )  # defauolt limit to top 50
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray().flatten()
    keyword_scores = sorted(
        zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True
    )

    return keyword_scores


def with_text_or_sg_rank(
    text: str, language: Language, algorithm: KeyphraseAlgorithm, limit=10
):
    corpus = textacy.Corpus(lang="en_core_web_sm")

    # Add the text to the corpus. The .add() method returns None,
    # but it modifies the 'corpus' object in-place.
    corpus.add(text)

    # Now, access the processed document from the corpus.
    # It will be the first (and only) document in the corpus.
    doc = corpus.docs[0]

    # Extract keyphrases using TextRank
    if algorithm == KeyphraseAlgorithm.TEXT_RANK:
        keyphrases = textacy.extract.keyterms.textrank(doc, topn=limit)
    # Extract keyphrases using SGRank
    else:
        keyphrases = textacy.extract.keyterms.sgrank(doc, topn=limit)

    return keyphrases
