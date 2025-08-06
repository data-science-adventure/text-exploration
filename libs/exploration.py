import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import nltk
from nltk.text import Text
from libs.util.text import get_stop_words
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from pathlib import Path
from model.context import Language
import libs.extraction as extraction

# Readability and Complexity Analysis
# Using Flesch-Kincaid Grade Level and Automated Readability Index (ARI)
# Requires average sentence length and average syllables per word.
# Syllable counting is complex; for simplicity, we'll use a rough estimate or external library if available.
# Textstat is a good external library, but we'll try to implement basic formulas.


def count_syllables_simple(word):
    # A very basic syllable counter (can be inaccurate for complex words)
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count = 1  # Every word has at least one syllable
    return count


def calculate_readability(text):
    sentences_list = sent_tokenize(text)
    words_list = word_tokenize(text)
    words_filtered_for_readability = [word for word in words_list if word.isalpha()]

    num_sentences = len(sentences_list)
    num_words = len(words_filtered_for_readability)
    if num_words == 0 or num_sentences == 0:
        return {
            "flesch_kincaid": 0,
            "ari": 0,
            "avg_sentence_length": 0,
            "avg_syllables_per_word": 0,
        }

    total_syllables = sum(
        count_syllables_simple(word) for word in words_filtered_for_readability
    )
    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = total_syllables / num_words

    # Flesch-Kincaid Grade Level: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    flesch_kincaid = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

    # Automated Readability Index (ARI): 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    # For ARI, we need character count.
    total_characters = sum(len(word) for word in words_filtered_for_readability)
    avg_chars_per_word = total_characters / num_words

    ari = 4.71 * avg_chars_per_word + 0.5 * avg_sentence_length - 21.43

    return {
        "flesch_kincaid": max(0, flesch_kincaid),  # Grade level shouldn't be negative
        "ari": max(0, ari),  # ARI shouldn't be negative
        "avg_sentence_length": avg_sentence_length,
        "avg_syllables_per_word": avg_syllables_per_word,
    }


def plot_frecuency_words(words):
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(20)
    df_most_common = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
    print(df_most_common)

    # Visualization: Bar Chart of Word Frequency
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=[word for word, freq in most_common_words],
        y=[freq for word, freq in most_common_words],
    )
    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Visualization: Word Cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Most Frequent Words")
    plt.show()


def plot_ngram(words):
    # N-gram Frequency (Bigrams and Trigrams)
    # Bigrams
    bigrams = list(ngrams(words, 2))
    bigram_freq = Counter(bigrams)
    most_common_bigrams = bigram_freq.most_common(20)

    df_most_common_bigrams = pd.DataFrame(
        most_common_bigrams, columns=["Bigram", "Frequency"]
    )
    df_most_common_bigrams["Bigram"] = df_most_common_bigrams["Bigram"].apply(
        lambda x: " ".join(x)
    )
    print(df_most_common_bigrams)

    # Visualization: Bar Chart of Top 20 Bigrams
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=[" ".join(bigram) for bigram, freq in most_common_bigrams],
        y=[freq for bigram, freq in most_common_bigrams],
    )
    plt.title("Top 20 Most Frequent Bigrams")
    plt.xlabel("Bigrams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Trigrams
    trigrams = list(ngrams(words, 3))
    trigram_freq = Counter(trigrams)
    most_common_trigrams = trigram_freq.most_common(20)

    df_most_common_trigrams = pd.DataFrame(
        most_common_trigrams, columns=["Trigram", "Frequency"]
    )
    df_most_common_trigrams["Trigram"] = df_most_common_trigrams["Trigram"].apply(
        lambda x: " ".join(x)
    )
    print(df_most_common_trigrams)

    # Visualization: Bar Chart of Top 20 Trigrams
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=[" ".join(trigram) for trigram, freq in most_common_trigrams],
        y=[freq for trigram, freq in most_common_trigrams],
    )
    plt.title("Top 20 Most Frequent Trigrams")
    plt.xlabel("Trigrams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_pos_tag_frequency(words):
    # POS Tagging
    pos_tags = nltk.pos_tag(
        words
    )  # Use 'words' (not filtered_words) to get POS for all words
    pos_tag_counts = Counter(tag for word, tag in pos_tags)
    most_common_pos_tags = pos_tag_counts.most_common(15)

    df_most_common_pos_tags = pd.DataFrame(
        most_common_pos_tags, columns=["Tag", "Frequency"]
    )
    print(df_most_common_pos_tags)

    # Visualization: Bar Chart of POS Tag Frequency
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=[tag for tag, freq in most_common_pos_tags],
        y=[freq for tag, freq in most_common_pos_tags],
    )
    plt.title("Most Frequent Part-of-Speech Tags")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# Concordance Analysis
def plot_concordance_analysis(words_to_track: list[str], words, language: Language):
    text_nltk = Text(words)
    for word in words_to_track:
        display_concordance(word, text_nltk)
        visualize_kwic_context(word, words, language)


def display_concordance(word, text_nltk, num_lines=10):
    """Displays concordance for a given word."""
    print(f"\n--- Concordance for '{word}' ---")
    text_nltk.concordance(word, lines=num_lines)


# Visualization: Simple "Concordance-like" bar chart for context words
def visualize_kwic_context(target_word, words, language, window_size=5, top_n=10):
    stop_words = get_stop_words(language)
    context_words = []
    for i, word in enumerate(words):
        if word == target_word:
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words.extend(
                [
                    w
                    for w in words[start:end]
                    if w != target_word and w not in stop_words
                ]
            )

    if not context_words:
        print(f"'{target_word}' not found in the text for context visualization.")
        return

    context_freq = Counter(context_words)
    most_common_context = context_freq.most_common(top_n)

    if not most_common_context:
        print(f"No common context words found for '{target_word}'.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=[word for word, freq in most_common_context],
        y=[freq for word, freq in most_common_context],
    )
    plt.title(f"Top {top_n} Context Words for '{target_word}'")
    plt.xlabel("Context Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def collocation_analysis(words):
    # Collocation Analysis
    # Find bigrams that are statistically significant collocations
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    bigrams = list(ngrams(words, 2))
    bigram_freq = Counter(bigrams)

    # Filter by frequency (e.g., only consider bigrams that appear at least 5 times)
    finder.apply_freq_filter(5)

    # Get the top N collocations based on a scoring function (e.g., PMI - Pointwise Mutual Information)
    # Other scores: nltk.collocations.likelihood_ratio, nltk.collocations.chi_sq
    top_collocations = finder.nbest(bigram_measures.pmi, 20)

    print("\n--- Top 20 Collocations (Bigrams) ---")
    for col in top_collocations:
        print(f"{' '.join(col)}")

    # Visualization: Bar Chart of Top Collocations (PMI score as an indicator)
    # Note: PMI scores can be negative or very large; normalizing or just showing the collocations is common.
    # For visualization, let's use a simple bar chart of their frequency, or if PMI is required,
    # we would need to extract and plot the PMI values.
    # For simplicity, we'll plot the frequency of the identified collocations.

    collocation_freqs = []
    for col in top_collocations:
        collocation_freqs.append((f"{col[0]} {col[1]}", bigram_freq[col]))

    collocation_freqs.sort(
        key=lambda x: x[1], reverse=True
    )  # Sort by frequency for plotting

    if collocation_freqs:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=[col[0] for col in collocation_freqs],
            y=[col[1] for col in collocation_freqs],
        )
        plt.title("Top 20 Collocations by Frequency (from PMI candidates)")
        plt.xlabel("Collocations")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print(
            "No significant collocations found for visualization based on current filters."
        )


from sklearn.feature_extraction.text import TfidfVectorizer


# Keyword Analysis (using TF-IDF for illustrative purposes, typically for multiple documents)
# For a single document, high TF-IDF indicates a word that is frequent in this document
# and relatively rare in a hypothetical corpus (or simply frequent in this one).

# TfidfVectorizer expects a list of documents, so we pass our combined text as a single-element list


def keyword_analysis(text: str, language: Language, max_features=50):
    vectorizer = TfidfVectorizer(
        stop_words=extraction.get_stop_words(language), max_features=max_features
    )  # defauolt limit to top 50
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray().flatten()
    keyword_scores = sorted(
        zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True
    )

    print("\n--- Top 20 Keywords (TF-IDF Scores) ---")
    for keyword, score in keyword_scores[:20]:
        print(f"{keyword}: {score:.4f}")

    # Visualization: Bar Chart of Top Keywords by TF-IDF Score
    keywords_to_plot = keyword_scores[:20]
    if keywords_to_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=[kw for kw, score in keywords_to_plot],
            y=[score for kw, score in keywords_to_plot],
        )
        plt.title("Top 20 Keywords by TF-IDF Score")
        plt.xlabel("Keywords")
        plt.ylabel("TF-IDF Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("No keywords found for visualization.")


def plot_word_dispersion(text_words, words_to_track):
    """Plots the dispersion of specified words throughout the text."""
    text_len = len(text_words)
    word_indices = {word: [] for word in words_to_track}

    for i, word in enumerate(text_words):
        if word in words_to_track:
            word_indices[word].append(i)

    plt.figure(figsize=(15, 7))
    for i, word in enumerate(words_to_track):
        y_coords = [i] * len(word_indices[word])
        plt.scatter(word_indices[word], y_coords, label=word, alpha=0.6)

    plt.yticks(range(len(words_to_track)), words_to_track)
    plt.xlabel("Word Offset (Position in Text)")
    plt.ylabel("Words")
    plt.title("Word Dispersion Plot")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def dispersion_analysis(words, words_to_track):
    # Ensure these words are present in the 'words' list (lowercase)
    words_to_track_filtered = [word for word in words_to_track if word in set(words)]

    if words_to_track_filtered:
        plot_word_dispersion(words, words_to_track_filtered)
    else:
        print(
            "None of the specified words were found in the text for dispersion analysis."
        )


def lexical_diversity_analysis(words, filtered_words):
    num_words = len(words)
    num_unique_words = len(set(words))
    num_filtered_words = len(filtered_words)
    num_unique_filtered_words = len(set(filtered_words))

    # Type-Token Ratio (TTR)
    ttr = num_unique_words / num_words if num_words > 0 else 0
    filtered_ttr = (
        num_unique_filtered_words / num_filtered_words if num_filtered_words > 0 else 0
    )

    data = {
        "Metric": [
            "Total Words (Tokens)",
            "Unique Words (Types)",
            "Type-Token Ratio (TTR)",
            "Filtered Words (Tokens - no stopwords/non-alphanumeric)",
            "Unique Filtered Words (Types)",
            "Filtered Type-Token Ratio (TTR)",
        ],
        "Value": [
            f"{num_words}",
            f"{num_unique_words}",
            f"{ttr:.4f}",
            f"{num_filtered_words}",
            f"{num_unique_filtered_words}",
            f"{filtered_ttr:.4f}",
        ],
    }
    print(pd.DataFrame(data))
    # Moving-Average Type-Token Ratio (MATTR) or Guiraud's Index for better comparison across texts
    # For simplicity, we'll focus on TTR and its filtered version.

    # Visualization: Bar Chart for TTR
    metrics = ["Total Words", "Unique Words", "Filtered Words", "Unique Filtered Words"]
    values = [
        num_words,
        num_unique_words,
        num_filtered_words,
        num_unique_filtered_words,
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=values)
    plt.title("Lexical Statistics")
    plt.ylabel("Count")
    plt.show()

    metrics_ttr = ["TTR (Raw)", "TTR (Filtered)"]
    values_ttr = [ttr, filtered_ttr]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics_ttr, y=values_ttr)
    plt.title("Type-Token Ratios")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.show()
