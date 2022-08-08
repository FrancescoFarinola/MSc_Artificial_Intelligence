"""
This file mainly contains functions to compute additional features,
predictions, answers and evaluation of the model
"""

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix
import gensim.downloader as gloader
import numpy as np
import pandas as pd
import tensorflow as tf
import preprocess
from preprocess import nlp
from collections import OrderedDict
import chars2vec
import re
from sklearn.feature_extraction.text import CountVectorizer
from settings import CHAR_EMBEDDING_DIM, BATCH_SIZE
from preprocess import preprocessing, expand_contractions, tokenization_spacy, split_alpha_num_sym, strip_text, \
    CHARS_TO_SPACE, CHARS_TO_REMOVE, spell_correction, lemmatization, lower

# list of possible POS and NER values
POS_LISTING = ["$", "``", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT",
               "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP",
               "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH",
               "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "_SP"]

NER_LISTING = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
               "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


def get_word_listing(sentences):
    """
    Compute word listing from list of sentences
    @param sentences: list containing sentences
    @return: word listing
    """
    terms = [term for sentence in sentences for term in sentence.split()]
    return list(set(terms))


def tokenize(word_listing):
    """
    Compute tokenizer, word_to_idx and idx_to_word
    @param word_listing: word listing
    @return: tokenizer, word_to_idx, idx_to_word
    """
    tokenizer = Tokenizer(filters=list(), oov_token=True)
    tokenizer.fit_on_texts(word_listing)
    indices = tokenizer.texts_to_sequences(word_listing)
    indices = [item for sublist in indices for item in sublist]
    word_to_idx = dict(zip(word_listing, indices))
    idx_to_word = dict(zip(indices, word_listing))

    return tokenizer, word_to_idx, idx_to_word


def get_co_occurrence_matrix(sentences, word_to_idx, window_size=4):
    """
    Compute co-occurrence matrix
    @param sentences: list of sentences
    @param word_to_idx: word_to_idx
    @param window_size: size of the window for the co-occurrence matrix
    @return: co-occurrence matrix
    """
    rows, cols, data = [], [], []
    for sentence in sentences:
        words = sentence.split()
        for index, word in enumerate(words):
            row = word_to_idx.get(word)
            lower_limit = max(0, index-window_size)
            upper_limit = min(len(words), index+window_size+1)
            for i in range(lower_limit, upper_limit):
                if i != index:
                    col = word_to_idx.get(words[i])
                    rows.append(row)
                    cols.append(col)
                    data.append(1)

    return csr_matrix((data, (rows, cols)))


def compute_oov_embeddings(terms, word_to_idx, idx_to_word, co_occurrence_matrix, embedding_dim, embedding_model,
                           random_strategy=False):
    """
    Compute embedding for OOV terms.
    By default, neighbour strategy is used.
    @param terms: word listing
    @param word_to_idx: word to idx
    @param idx_to_word: idx to word
    @param co_occurrence_matrix: co-occurrence matrix
    @param embedding_dim: word embedding dimension
    @param embedding_model: embedding model
    @param random_strategy: apply (or not) random strategy
    @return: embeddings for oov terms
    """
    embeddings = dict()
    vocabulary = embedding_model.key_to_index.keys()

    for term in terms:
        # if random strategy is adopted, get a random vector
        if random_strategy:
            embeddings[term] = np.random.rand(embedding_dim)
        else:
            # the embedding of an oov is the average of the embeddings
            # of its neighbors
            count = 0
            s = np.zeros(embedding_dim)
            i = word_to_idx.get(term)
            co_occ_vec = co_occurrence_matrix.getrow(i)
            cols, values = co_occ_vec.indices, co_occ_vec.data
            for j in range(len(cols)):
                col, value = cols[j], values[j]
                if i != col:
                    neighbor = idx_to_word.get(col)
                    if neighbor in vocabulary:
                        count += value
                        s += embedding_model.get_vector(neighbor) * value
            # if no neighbors, get a random vector
            if count == 0:
                embeddings[term] = np.random.rand(embedding_dim)
            else:
                embeddings[term] = s / count
    return embeddings


def get_embedding_matrix(dataframe, embedding_dim):
    """
    Compute word embedding matrix
    @param dataframe: dataframe
    @param embedding_dim: word embedding dimension
    @return: word embedding matrix
    """

    # compute word listing, tokenizer, word2idx, idx2word
    all_text = pd.concat([dataframe['context'], dataframe['question']], axis=0).unique()  #
    df_word_listing = get_word_listing(all_text)

    df_tokenizer, df_word_to_idx, df_idx_to_word = tokenize(df_word_listing)
    df_co_occurrence_matrix = get_co_occurrence_matrix(all_text, df_word_to_idx)

    # load glove embedding model
    print("Loading GloVe embedding model...")
    embedding_model = gloader.load("glove-wiki-gigaword-{}".format(embedding_dim))

    # dictionary word -> embedding vector
    embedding_dic = {key: embedding_model.get_vector(key)
                     for key in set(df_word_listing).intersection(embedding_model.key_to_index.keys())}
    print(f"There are {len(embedding_dic)} words for which we already know the embedding")
    oov = set(df_word_listing) - embedding_dic.keys()
    print(f"There are {len(oov)} oov words")

    # compute embedding for oov
    print("Computing out-of-vocabulary embeddings...")
    embeddings_oov = compute_oov_embeddings(oov, df_word_to_idx, df_idx_to_word, df_co_occurrence_matrix,
                                            embedding_dim, embedding_model)
    # join the two embedding dictionaries
    embedding_dic = {**embedding_dic, **embeddings_oov}

    # compute embedding matrix
    print("Computing embedding matrix...")
    emb_matrix = np.zeros((len(embedding_dic) + 2, embedding_dim))  # one for padding PAD and one for unknown UNK
    for k, v in embedding_dic.items():
        idx = df_word_to_idx.get(k)
        emb_matrix[idx] = v

    # return embedding matrix, word listing, tokenizer, word2idx, idx2word
    return emb_matrix, df_word_listing, df_tokenizer, df_word_to_idx, df_idx_to_word


def get_max_length(dataframe, rate=1.1):
    """
    Compute maximum length for contexts, texts and questions
    @param dataframe: dataframe
    @param rate: how much enlarge the maximum dimension
    @return: maximum context, text and question length
    """
    # compute maximum length of contexts
    len_context_tokens = [len(sentence.split()) for sentence in dataframe.context.unique()]
    max_context_length = np.max(len_context_tokens)
    print(f'Max length for context is {max_context_length}')
    print(f'Max length adopted for context is {int(max_context_length * rate)}')

    # compute maximum length of answes
    len_text_tokens = [len(sentence.split()) for sentence in dataframe.text.values]
    max_text_length = np.max(len_text_tokens)
    print(f'Max length for answer is {max_text_length}')
    print(f'Max length adopted for answer is {int(max_text_length * rate)}')

    # compute maximum length of questions
    len_question_tokens = [len(sentence.split()) for sentence in dataframe.question.values]
    max_question_length = np.max(len_question_tokens)
    print(f'Max length for question is {max_question_length}')
    print(f'Max length adopted for question is {int(max_question_length * rate)}')

    # return the maximum length for context, answers and questions
    return int(max_context_length * rate), int(max_text_length * rate), int(max_question_length * rate)


def pad(df_values, tokenizer, max_length):
    """
    Pad records
    @param df_values: list of records to pad
    @param tokenizer: tokenizer
    @param max_length: maximum length for padding
    @return: padded records
    """
    x = [t.split() for t in df_values]
    x_encoded = tokenizer.texts_to_sequences(x)
    # tokenizer returns None for oov, here None is replaced with index 1
    x_encoded = [[1 if i is None else i for i in row] for row in x_encoded]
    # pad sequences up to the desired maximum length at the end (='post')
    x_padded = pad_sequences(x_encoded, maxlen=max_length, padding='post')
    # return padded records
    return x_padded


def compute_tf(df, max_context_length):
    """
    Compute term frequency
    @param df: dataframe
    @param max_context_length: maximum context length
    @return: context term frequencies
    """
    print("Computing TF...")
    corpus = df.context.values

    # token pattern is 'everything but not a whitespace'
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    tf_context = vectorizer.fit_transform(corpus)

    # substitute each term with its frequency
    # (apply on the unique contexts, then map them)
    tmp = pd.DataFrame(df.context.unique(), columns=['context'])
    tfs = []
    for i, row in tmp.context.iteritems():
        tokens = row.split()
        tfs.append([tf_context[i, vectorizer.vocabulary_[token]] for token in tokens])

    # map from unique context to all contexts in df
    dict_context_tf = dict(zip(tmp.context, tfs))
    df_tf = df.context.apply(lambda x: dict_context_tf.get(x))

    # pad records and return them
    df_tf_padded = pad_sequences(df_tf, maxlen=max_context_length, padding='post', truncating='post')
    return df_tf_padded


# Exact match
def exact_match(df, max_context_length):
    """
    Compute exact match between context and question
    @param df: dataframe
    @param max_context_length: maximum context length
    @return: dataframe with exact match
    """
    match = []
    for i in range(0, df.shape[0]):
        # array containing 1 if the context term is in the question, 0 otherwise
        match1 = np.in1d(df.context[i].split(), df.question[i].split()).astype(int).reshape(1, -1)
        # pad record
        padded_match = pad_sequences(match1, padding="post", value=0, maxlen=max_context_length, truncating='post')
        match.append(padded_match)
    return np.array(match)


def apply_exact_match(df, pipeline, max_context_length):
    """
    Apply exact match
    @param df: dataframe
    @param pipeline: list of preprocessing operations
    @param max_context_length: maximum context length
    @return: dataframe with exact match
    """
    df2 = df.copy()
    # apply preprocessing before compute matching
    df2, _ = preprocess.apply_preprocessing(df2, pipeline, text=False)
    # remove stopwords from question before computing exact match
    df2['question'] = df2['question'].apply(lambda x: preprocess.remove_stopwords(x))
    return exact_match(df2, max_context_length).squeeze()


def compute_exact_match(df, max_context_length):
    """
    Compute original, lowercase and lemmatized exact matches
    @param df: dataframe
    @param max_context_length: maximum context length
    @return: dataframe with exact matches
    """

    print("Computing original exact match...")
    preprocessing_pipeline1 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.strip_text]

    original_match = apply_exact_match(df, preprocessing_pipeline1, max_context_length).squeeze()

    print("Computing lowercase exact match...")
    preprocessing_pipeline2 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.lower,
                               preprocess.strip_text]

    lowercase_match = apply_exact_match(df, preprocessing_pipeline2, max_context_length).squeeze()

    print("Computing lemmatized exact match...")
    preprocessing_pipeline3 = [preprocess.expand_contractions,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.lemmatization,
                               preprocess.lower,
                               preprocess.strip_text]

    lemmatized_match = apply_exact_match(df, preprocessing_pipeline3, max_context_length).squeeze()

    # stack the 3 exact matches
    exact_match_input = np.stack((original_match, lowercase_match, lemmatized_match), axis=-1).astype(np.float32)
    return exact_match_input


# POS tags
def create_pos_dicts(pos_listing=POS_LISTING):
    """
    Compute pos to idx and idx to pos
    @param pos_listing: list of POS tags
    @return: pos to idx, idx to pos
    """
    print("Creating dictionaries for POS tags...")
    pos2idx = OrderedDict({tag: idx for idx, tag in enumerate(pos_listing)})
    idx2pos = OrderedDict({idx: tag for tag, idx in pos2idx.items()})
    # inserting pad token with idx=0 and moving first one to last
    pos2idx.update({(list(pos2idx.keys()))[0]: len(pos2idx)})
    idx2pos.update({len(idx2pos): (list(idx2pos.values()))[0]})
    pos2idx.move_to_end((list(pos2idx.keys()))[0], last=True)
    idx2pos.move_to_end(0, last=True)
    pos2idx.update({'<PAD>': 0})
    idx2pos.update({0: '<PAD>'})
    pos2idx.move_to_end('<PAD>', last=False)
    idx2pos.move_to_end(0, last=False)
    pos_listing = list(pos_listing)
    pos_listing.append(pos_listing.pop(0))
    pos_listing.insert(0, '<PAD>')
    return pos2idx, idx2pos


def compute_pos(df, contexts, tag2idx, max_context_length):
    """
    Compute POS
    @param df: dataframe
    @param contexts: unique contexts
    @param tag2idx: pos to idx
    @param max_context_length: maximum context length
    @return: pos dataframe
    """
    print("Computing POS tags...")
    docs = nlp.pipe(contexts.context, disable=["ner", "lemmatizer"])
    postags = [[token.tag_ for token in doc] for doc in docs]

    # convert to integers using dict
    indexed_pos = [[tag2idx[tag] for tag in context] for context in postags]

    # pad
    print("Padding POS sequences...")
    padded_pos = pad_sequences(indexed_pos, padding="post", value=tag2idx['<PAD>'],
                               maxlen=max_context_length, truncating='post')

    # mapping from unique contexts to all contexts
    dict_pos = dict(zip(contexts.context, padded_pos))
    pos_tmp = df.context.apply(lambda x: dict_pos.get(x))
    pos = np.array([t for t in pos_tmp])
    return pos


# NER tags
def create_ner_dicts(ner_listing=NER_LISTING):
    """
    Compute ner to idx and idx to ner
    @param ner_listing: list of NER tags
    @return: ner to idx, idx to ner
    """
    print("Creating dictionaries for NER tags...")
    ner2idx = OrderedDict({tag: idx for idx, tag in enumerate(ner_listing)})
    idx2ner = OrderedDict({idx: tag for tag, idx in ner2idx.items()})
    # inserting pad token with idx=0 and moving first one to last
    ner2idx.update({(list(ner2idx.keys()))[0]: len(ner2idx)})
    idx2ner.update({len(idx2ner): (list(idx2ner.values()))[0]})
    ner2idx.move_to_end((list(ner2idx.keys()))[0], last=True)
    idx2ner.move_to_end(0, last=True)
    ner2idx.update({'<PAD>': 0})
    idx2ner.update({0: '<PAD>'})
    ner2idx.move_to_end('<PAD>', last=False)
    idx2ner.move_to_end(0, last=False)
    ner_listing = list(ner_listing)
    ner_listing.append(ner_listing.pop(0))
    ner_listing.insert(0, '<PAD>')
    # insert none token for words without NER tag
    ner_listing.append('NONE')
    ner2idx.update({'NONE': len(ner2idx)})
    idx2ner.update({len(idx2ner): 'NONE'})
    return ner2idx, idx2ner


def chars_to_space(text):
    """
    replace chars in text with a space
    @param text: input text
    @return: output text
    """
    return CHARS_TO_SPACE.sub(' ', text)


def chars_to_remove(text):
    """
    remove chars from text
    @param text: input text
    @return: output text
    """
    return CHARS_TO_REMOVE.sub('', text)


def compute_ner(df1, contexts_df1, contexts, ner2idx, max_context_length, debug=False):
    """
    Compute NER
    Since applying the entire preprocessing undermine correct named entity recognition,
    first it is applied only some part of preprocessing, then ner is computed,
    after that remaining preprocessing is applied.

    @param df1: dataframe preprocessed with pipeline 1
    @param contexts_df1: unique contexts from dataframe preprocessed with pipeline 1
    @param contexts: unique contexts from original dataframe
    @param ner2idx: ner to idx
    @param max_context_length: maximum context length
    @param debug: True if return also nertags of type string
    @return: ner dataframe
    """
    print("Computing NER tags...")
    # apply some preprocessing from PREPROCESSING_PIPELINE1
    pipeline = [expand_contractions, tokenization_spacy, chars_to_space, split_alpha_num_sym, strip_text]
    unique_contexts_df = contexts.copy()
    unique_contexts_df.context = unique_contexts_df.context.apply(lambda x: preprocessing(x, pipeline))
    docs = nlp.pipe(unique_contexts_df.context.values, disable=["tok2vec", "tagger", "lemmatizer"])
    # compute ner
    nertags = [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]
    indexed_ner = []

    # convert to integers using dict
    for i in range(len(nertags)):
        # k is reset (k=0) at each row
        k = 0
        splits = contexts_df1.context[i].split()
        # all ner tags in sentences are equal to the none tag
        sentence = np.full(shape=(len(splits)), fill_value=ner2idx['NONE'])
        # for each tuple (words, tag)
        for words, tag in nertags[i]:
            # apply remaining preprocessing from PREPROCESSING_PIPELINE1
            words = strip_text(lower(lemmatization(spell_correction(split_alpha_num_sym(chars_to_remove(words))))))
            # match word with its tag
            for word in words.split():
                k = k + splits[k:].index(word)
                sentence[k] = ner2idx[tag]
                k += 1
        indexed_ner.append(sentence)
    indexed_ner = np.array(indexed_ner, dtype=object)

    # padding
    print("Padding NER sequences...")
    padded_ner = pad_sequences(indexed_ner, padding="post", value=ner2idx['<PAD>'],
                               maxlen=max_context_length, truncating='post')

    # mapping from unique context to all contexts
    dict_ner = dict(zip(contexts_df1.context, padded_ner))
    ner_tmp = df1.context.apply(lambda x: dict_ner.get(x))
    ner = np.array([t.tolist() for t in ner_tmp])

    # if true, return additional information
    if debug:
        return ner, nertags
    return ner


def get_char_embeddings(word_listing, word_to_idx):
    """
    Compute character embedding matrix
    @param word_listing: word listing
    @param word_to_idx: word to idx
    @return: character embedding matrix
    """
    print("Computing character-level embeddings...")
    c2v_model = chars2vec.load_model(f'eng_{CHAR_EMBEDDING_DIM}')
    char_embs = c2v_model.vectorize_words(word_listing)

    # embedding dictionary
    char_emb_dict = dict(zip(word_listing, char_embs))

    #embedding matrix
    char_embedding_matrix = np.zeros((len(char_emb_dict) + 2, CHAR_EMBEDDING_DIM))  # +1 per il padding +2 per l'UNK
    for k, v in char_emb_dict.items():
        idx = word_to_idx.get(k)
        char_embedding_matrix[idx] = v

    return char_embedding_matrix


def compute_answers(predictions, df, df2):
    """
    Compute the answers given the indices span predictions.
    It reconstructs the original answer by using an intermediate dataset
    @param predictions: predictions (probabilities)
    @param df: original dataframe
    @param df2: dataframe processed with pipeline 2
    @return: answers extracted from the original dataset
    """
    # get predicted start and end indices
    preds = np.argmax(predictions, -1)
    s_idx = preds[0]
    e_idx = preds[1]

    spans = []
    for i in range(preds.shape[1]):
        # get the span from the intermediate
        r2 = df2.loc[i].context.split()[s_idx[i]:e_idx[i]+1]
        # there may be symbols or spaces between words, only symbols between characters inside the same word
        s = r'[^\w£$%]*?'.join([r'[^\w\s£$%]*?'.join([re.escape(ch) for ch in word]) for word in r2])
        # how many tokens differ the same row in the two datasets
        a = abs(len(df2.loc[i].context.split()) - len(df.loc[i].context.split()))
        # idx where the span starts
        idx = len(' '.join(df2.loc[i].context.split()[:s_idx[i]]))
        # subtract the difference of lenght between intermediate and original row from index
        # and find a matching in the original dataframe row
        xre = re.search(s, df.context[i][max(0, idx-a):])
        spans.append(xre.group())
    return spans


def computing_predictions(model, df, x, batch_size):
    """
    Compute predictions and answers
    @param model: model
    @param df: dataframe
    @param x: input to the model
    @param batch_size: batch size
    @return: answers extracted from original dataframe
    """
    print("Preprocessing on datasets...")
    print("Applying expand_contractions2, tokenization_spacy, remove_chars, split_alpha_num_sym and strip_text.")
    preprocessing_pipeline_ = [preprocess.expand_contractions2,
                               preprocess.tokenization_spacy,
                               preprocess.remove_chars,
                               preprocess.split_alpha_num_sym,
                               preprocess.strip_text]
    df2 = df.copy()
    df2, tmp2 = preprocess.apply_preprocessing(df2, preprocessing_pipeline_, text=False)
    print("Calculating predictions...")
    predictions = model.predict(x, batch_size=batch_size)
    print("Computing answers...")
    spans = compute_answers(predictions, df, df2)
    data = dict(zip(df.id, spans))
    return data


def evaluate_model(model, max_context_length, truth_df, x, batch_size=BATCH_SIZE):
    """
    Compute evaluation by using the indices of the relevant/retrieved tokens
    Compute average precision, recall and f1 score.

    @param model: model
    @param max_context_length: maximum context length
    @param truth_df: ground truth dataframe (preprocessed 1)
    @param x: input to the model
    @param batch_size: batch size
    @return: F1, precision, recall
    """
    print("Computing F1 score, precision and recall...")

    r = np.arange(max_context_length)

    # create truth_mask
    # truth mask: 1 if (truth start <= r <= truth end), 0 otherwise
    truth_mask = (truth_df.s_idx.values[:, None] <= r) & (r <= truth_df.e_idx.values[:, None])
    truth_mask = truth_mask.astype('int8')

    # create predictions mask
    # get predicted start / end
    predictions_start, predictions_end = model.predict(x, batch_size=batch_size)
    predicted_s_idx, predicted_e_idx = np.argmax(predictions_start, -1), np.argmax(predictions_end, -1)
    # prediction mask: 1 if (predicted start <= r <= predicted end), 0 otherwise
    predictions_mask = (predicted_s_idx[:, None] <= r) & (r <= predicted_e_idx[:, None])
    predictions_mask = predictions_mask.astype('int8')

    # shared mask: element-wise multiplication between truth mask and predictions mask
    shared_mask = tf.math.multiply(truth_mask, predictions_mask)

    # number of shared indices (relevant and retrieved)
    shared = np.sum(shared_mask, axis=-1)
    # number of retrieved indices
    predictions = np.sum(predictions_mask, axis=-1)
    # number of relevant indices
    truth = np.sum(truth_mask, axis=-1)

    # precision: (relevant and retrieved) / retrieved
    precision = shared / predictions
    # recall: (relevant and retrieved) / relevant
    recall = [shared[i] / truth[i] if truth[i] else 0 for i in range(truth_df.shape[0])]

    # f1 score
    f1_score = [2 * precision[i] * recall[i] / (precision[i] + recall[i])
                if precision[i] + recall[i] else 0 for i in range(truth_df.shape[0])]

    return np.average(f1_score), np.average(precision), np.average(recall), shared_mask, predictions_mask, truth_mask
