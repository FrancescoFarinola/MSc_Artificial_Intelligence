"""
Our model.
"""


import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, \
    Dense, GRU, Input, Concatenate, Softmax, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from settings import CHAR_EMBEDDING_DIM


class SimilarityLayer(layers.Layer):
    """
    Return the similarity matrix between contexts and questions
    """
    def __init__(self, max_question_length, max_context_length, **kwargs):
        super(SimilarityLayer, self).__init__(**kwargs)
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length

    def build(self, input_shape):
        # w is a trainable weight vector
        self.w = self.add_weight(shape=(input_shape[0][-1] * 3, 1), trainable=True, name='sim_w')

    def call(self, inputs, **kwargs):
        # H is the contextual embedding of the contexts, U of the questions
        H, U = inputs
        # axes are: (batch, context, question, embedding)
        # tile H on the question axis and
        # tile U on the context axis
        H_dim_repeat = [1, 1, self.max_question_length, 1]
        U_dim_repeat = [1, self.max_context_length, 1, 1]
        repeated_H = K.tile(K.expand_dims(H, axis=2), H_dim_repeat)
        repeated_U = K.tile(K.expand_dims(U, axis=1), U_dim_repeat)
        # now that both H and U have dimension (batch, context, question, embedding)
        # compute element wise multiplication between contexts and questions
        element_wise_multiply = repeated_H * repeated_U
        # concatenate H, U and the product
        concatenated_tensor = K.concatenate(
            [repeated_H, repeated_U, element_wise_multiply], axis=-1)
        # multiply to the trainable weight
        # squeeze on embedding axis to get tensor of dimension (batch, context, question)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.w), axis=-1)
        # return similarity matrix
        return dot_product

    def get_config(self):
        config = super(SimilarityLayer, self).get_config()
        config.update({"max_question_length": self.max_question_length,
                       "max_context_length": self.max_context_length})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class C2Q(layers.Layer):
    """
    Context to Query
    """
    def __init__(self, **kwargs):
        super(C2Q, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # U is the contextual embedding of the questions,
        # S is the similarity matrix between contexts and questions
        U, S = inputs
        # get attention weighs by applying softmax to S
        a = Softmax(name='a')(S)
        # add context axis to U to get (batch, context, question, embedding)
        U2 = K.expand_dims(U, axis=1)
        # add 'embedding' axis to a to get (batch, context, question, 1)
        # multiply a and U
        # sum over question axis to get (batch, context, embedding)
        return K.sum(K.expand_dims(a, axis=-1) * U2, -2)

    def get_config(self):
        config = super(C2Q, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Q2C(layers.Layer):
    """
    Query to Context
    """
    def __init__(self, max_context_length, **kwargs):
        super(Q2C, self).__init__(**kwargs)
        self.max_context_length = max_context_length

    def call(self, inputs, **kwargs):
        # H is the contextual embedding of the context,
        # S is the similarity matrix
        H, S = inputs
        # compute max on last axis
        # compute softmax to get attention weights b
        b = Softmax(name='b')(K.max(S, axis=-1))
        # expand d on last axis to get (batch, context, 1)
        b = tf.expand_dims(b, -1)
        # multiply b * H and sum over context axis
        h_ = K.sum(b * H, -2)
        # add context axis
        h_2 = K.expand_dims(h_, 1)
        # tile on context axis
        # return matrix of (batch, context, embedding)
        return K.tile(h_2, [1, self.max_context_length, 1])

    def get_config(self):
        config = super(Q2C, self).get_config()
        config.update({"max_context_length": self.max_context_length})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MergeG(layers.Layer):
    def __init__(self, **kwargs):
        super(MergeG, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # H is the contextual embedding of the context
        # U_, H_ are C2Q and Q2C
        # em, pos, ner, tf are exact matching, POS, NER, term frequency
        H, U_, H_, em, pos, ner, tf = inputs
        # multiply H with C2Q
        HU_ = H * U_
        # multiply H with Q2C
        HH_ = H * H_
        # concatenate and return
        return K.concatenate([H, U_, HU_, HH_, em, pos, ner, tf])

    def get_config(self):
        config = super(MergeG, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Prediction(layers.Layer):
    def __init__(self, **kwargs):
        super(Prediction, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # s,e are start/end index probabilities of size (batch, context)
        s, e = inputs
        # add one axis to s, and one to e
        s = tf.expand_dims(s, axis=2)  #(batch, context, 1)
        e = tf.expand_dims(e, axis=1)  #(batch, 1, context)
        # matrix multiplication: get a squared matrix
        # of (batch, context, context)
        outer = tf.matmul(s, e)
        # put everything outside the central band to 0
        # the band has width 15
        outer = tf.linalg.band_part(outer, 0, 15)
        return outer

    def get_config(self):
        config = super(Prediction, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(max_question_length, max_context_length, embedding_dim, embedding_matrix, char_embedding_matrix,
                pos_embedding_matrix, ner_embedding_matrix):
    VOCAB_SIZE = embedding_matrix.shape[0]
    units = 50
    # inputs
    input_question = Input(shape=(max_question_length,), dtype='int32', name='question')
    input_context = Input(shape=(max_context_length,), dtype='int32', name='context')
    input_em = Input(shape=(max_context_length, 3), dtype='float32', name='em')
    input_pos = Input(shape=(max_context_length,), dtype='int32', name='pos')
    input_ner = Input(shape=(max_context_length,), dtype='int32', name='ner')
    input_tf = Input(shape=(max_context_length, 1), dtype='float32', name='tf')

    # encodings
    question_encoding = Embedding(VOCAB_SIZE, embedding_dim, trainable=False,
                                  input_length=max_question_length, mask_zero=True,
                                  embeddings_initializer=Constant(embedding_matrix),
                                  name='q_encoding')(input_question)

    paragraph_encoding = Embedding(VOCAB_SIZE, embedding_dim, trainable=False,
                                   input_length=max_context_length, mask_zero=True,
                                   embeddings_initializer=Constant(embedding_matrix),
                                   name='p_encoding')(input_context)

    pos_encoding = Embedding(pos_embedding_matrix.shape[0], pos_embedding_matrix.shape[1], trainable=False,
                             input_length=max_context_length, mask_zero=True,
                             embeddings_initializer=Constant(pos_embedding_matrix),
                             name='pos_encoding')(input_pos)

    ner_encoding = Embedding(ner_embedding_matrix.shape[0], ner_embedding_matrix.shape[1], trainable=False,
                             input_length=max_context_length, mask_zero=True,
                             embeddings_initializer=Constant(ner_embedding_matrix),
                             name='ner_encoding')(input_ner)

    char_question_encoding = Embedding(VOCAB_SIZE, CHAR_EMBEDDING_DIM, trainable=False,
                                       input_length=max_question_length, mask_zero=True,
                                       embeddings_initializer=Constant(char_embedding_matrix),
                                       name='char_q_encoding')(input_question)

    char_paragraph_encoding = Embedding(VOCAB_SIZE, CHAR_EMBEDDING_DIM, trainable=False,
                                        input_length=max_context_length, mask_zero=True,
                                        embeddings_initializer=Constant(char_embedding_matrix),
                                        name='char_p_encoding')(input_context)

    # concatenate context word and char embedding, and pass to dense
    p = Concatenate(-1, name='concat_p')([paragraph_encoding, char_paragraph_encoding])
    p2 = Dense(2*units, 'relu', name='dense_p')(p)

    # concatenate question word and char embedding and pass to dense
    q = Concatenate(-1, name='concat_q')([question_encoding, char_question_encoding])
    q2 = Dense(2*units, 'relu', name='dense_q')(q)

    # P rnn to get contextual embedding of the context
    H = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='H'), name='biH')(p2)

    # Q rnn to get contextual embedding of the question
    U = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='U'), name='biU')(q2)

    # get similarity matrix
    S = SimilarityLayer(max_question_length, max_context_length, name='S')([H, U])

    # get Context to Query
    U_ = C2Q(name='C2Q')([U, S])

    # get Query to Context
    H_ = Q2C(max_context_length, name='Q2C')([H, S])

    # merge
    G = MergeG(name='G')([H, U_, H_, input_em, pos_encoding, ner_encoding, input_tf])

    # first modeling layer
    M = Bidirectional(GRU(units, return_sequences=True, dropout=0.2), name='M')(G)

    # concatenate G and M
    GM = Concatenate(name='GM')([G, M])
    start = layers.TimeDistributed(Dense(1, name='dense_s'), name='td_s')(GM)
    # start index probabilities
    start = Softmax(name='start_')(tf.squeeze(start, -1))

    # second modeling layer
    M2 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2), name='M2')(M)
    # concatenate G and M2
    GM2 = Concatenate(name='GM2')([G, M2])
    end = layers.TimeDistributed(Dense(1, name='dense_e'), name='td_e')(GM2)
    # end index probabilities
    end = Softmax(name='end_')(tf.squeeze(end, -1))

    # start and end indices (max distance 15)
    outer = Prediction(name='prediction')([start, end])
    # get start indices
    start_pos = Lambda(lambda x: tf.reduce_max(x, axis=2), name='start')(outer)
    # get end indices
    end_pos = Lambda(lambda x: tf.reduce_max(x, axis=1), name='end')(outer)

    # model
    model = Model([input_context, input_question, input_em, input_pos, input_ner, input_tf],
                  [start_pos, end_pos])
    return model
