# Dante's text generation

This is an additional project work for the Natural Language Processing course of the MSc in Artificial Intelligence at Alma Mater Studiorum - University of Bologna.
The aim is to exploit and try to implement different techniques for text generation using Dante's Divine Comedy as dataset.

Among all the techniques seen, the one working better is the char-level text generation, so it might be a good idea to focus on improvements of this model.
Seq2Seq model is the one working worse since old italian word embeddings might be hard to learn. A lot of modifications to the model have been tried but none of them gave significantly better results, so i'll leave the base model.


**Things tried and implemented:**

1. Methods:
    - Seq2Seq text generation
    - Markov Chain text generation
    - Char-level text generation (best)
2. Sampling techniques:
    - Greedy Search
    - Temperature sampling
    - Top-k sampling
    - Beam search
3. Scoring:
    - BLEU score
    - METEOR score
4. Tries on seq2seq model:
    - Random strategy for word embeddings
    - Similarity strategy for word embeddings
    - Substituted two stacked GRUs in Encoder with a Bidirectional GRU
    - Added Bahdanau and Luong Attention (AdditiveAttention and Attention keras layers)
    - Added normalized term frequency as input to the encoder and decoder (both and individually - 3 tries) to see if unfrequent terms are handled better
    - Different batch sizes (even 1), stateful and stateless GRUs, number of units, dropout
