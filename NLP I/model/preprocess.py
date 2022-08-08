"""
Functions used to apply preprocessing operations to data
"""

import pandas as pd
import nltk
import spacy
from symspellpy import SymSpell, Verbosity
import pkg_resources
import re
from functools import reduce
from nltk.stem import WordNetLemmatizer

sym_spell = SymSpell(max_dictionary_edit_distance=2)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, 0, 1)

nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'senter', 'attribute_ruler'])
STOPWORDS = nlp.Defaults.stop_words

wordnet_lemmatizer = WordNetLemmatizer()

WHITESPACES_RE = re.compile(r"\s+")
CHARS_TO_SPACE = re.compile(r"[–—\-\\/\[\]()+:]")
CHARS_TO_REMOVE = re.compile(r"[^\w\s£$%]")


def expand_contractions(text):
    """
    Replace contracted forms with their non-contracted form
    @param text: text with contractions
    @return: text without contractions
    """
    text = re.sub(r"won't\b", "will not", text)
    text = re.sub(r"can't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'s\b", " s", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'m\b", " am", text)

    # string operation
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')

    # return text without contractions
    return text


def expand_contractions2(text):
    """
    Split contractions in two tokens without changing any character
    @param text: text with contractions
    @return: expanded text
    """
    text = re.sub(r"won't\b", "wo n't", text)
    text = re.sub(r"can't\b", "ca n't", text)
    text = re.sub(r"n't\b", " n't", text)
    text = re.sub(r"'re\b", " 're", text)
    text = re.sub(r"'s\b", " 's", text)
    text = re.sub(r"'d\b", " 'd", text)
    text = re.sub(r"'ll\b", " 'll", text)
    text = re.sub(r"'ve\b", " 've", text)
    text = re.sub(r"'m\b", " 'm", text)

    # string operation
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')

    # return text with splitted contractions
    return text


def tokenization_spacy(text):
    """
    Compute tokenization by using spacy
    @param text: original text
    @return: tokenized text
    """
    return ' '.join([token.text for token in nlp(text, disable=["tagger", "ner", "lemmatizer"])])


def remove_chars(text):
    """
    Remove selected characters or replace them with a space
    @param text: original text
    @return: text with selected characters removed or replaced by space
    """
    text = CHARS_TO_SPACE.sub(' ', text)  # split words
    return CHARS_TO_REMOVE.sub('', text)  # do not split words


def split_alpha_num_sym(text):
    """
    Split alphabetic characters, digits and symbols
    @param text: original text
    @return: split text
    """
    # split alphabetic from numeric characters and symbols and vice-versa
    text = re.sub(r'(\d)([a-zA-Z£$%])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z£$%])(\d)', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([£$%])', r'\1 \2', text)
    text = re.sub(r'([£$%])([a-zA-Z])', r'\1 \2', text)
    return text


def spell_correction(text):
    """
    Spell correction with edit distance up to 2
    Numeric, titles and words whose length < 5 are not changed
    @param text: original text
    @return: corrected text
    """
    # max edit distance: 2
    results = [t if (t.isnumeric() or t.istitle() or len(t) < 5)
               else sym_spell.lookup(t, Verbosity.TOP, max_edit_distance=2,
                                     include_unknown=True)[0].term
               for t in text.split()]
    return ' '.join(results)


def lemmatization(text):
    """
    Lemmatization
    @param text: original text
    @return: lemmatized text
    """
    return ' '.join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])


def lower(text):
    """
    Lowercase the text
    @param text: original text
    @return: text in lowercase
    """
    return text.lower()


def strip_text(text):
    """
    Replace multiple whitespaces with a single whitespace
    @param text: original text
    @return: stripped text
    """
    return WHITESPACES_RE.sub(' ', text)


def remove_stopwords(text, stopwords=STOPWORDS):
    """
    Remove stopwords from text
    @param text: original text
    @param stopwords: stopwords list
    @return: text without stopwords
    """
    return " ".join([w for w in text.split() if not (w in stopwords)])


def preprocessing(text, preprocessing_pipeline):
    """
    Apply preprocessing to text
    @param text: original text
    @param preprocessing_pipeline: list of preprocessing operations
    @return: preprocessed text
    """
    return reduce(lambda x, f: f(x), preprocessing_pipeline, text)


def apply_preprocessing(df, pipeline, text=True):
    """
    Apply preprocessing to dataset
    @param df: dataset
    @param pipeline: list of preprocessing operations
    @param text: apply (or not) preprocessing to 'text' column
    @return: preprocessed dataset,
             dataset containing only distinct contexts
    """
    # get distinct contexts
    unique_contexts = pd.DataFrame(df.context.unique(), columns=['context'])
    # apply preprocessing on distinct contexts
    unique_contexts.context = unique_contexts.context.apply(lambda x: preprocessing(x, pipeline))
    # mapping:  not_preprocessed_context -> preprocessed_context
    dict_context = dict(zip(df.context.unique(), unique_contexts.context))
    # substitute not_preprocessed_context with preprocessed_context
    df.context = df.context.apply(lambda x: dict_context.get(x))

    # if text is True apply preprocessing also on 'text' column
    if text:
        df['text'] = df['text'].apply(lambda x: preprocessing(x, pipeline))

    df['question'] = df['question'].apply(lambda x: preprocessing(x, pipeline))
    return df, unique_contexts
