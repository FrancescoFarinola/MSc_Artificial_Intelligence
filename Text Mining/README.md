# Text mining project work
This is a project work for the module Text Mining of the course of Data mining, Text mining and Big Data Analytics in the MSc of Artificial Intelligence at the University of Bologna.

[Colab notebook](https://colab.research.google.com/drive/1Sg-3LyKW1nMqUPfh4Mw2_YvZxdK5FGCs?usp=sharing)

The purpose of this project is a first approach to the programming language R by exploiting text mining techniques and different ML-DL models for text classification.

We will use the Reuters news dataset.

In the first part, we will investigate the dataset by looking at the class distribution and we will simplify the dataset by keeping only the classes with similar cardinality - this will prevent us from applying oversampling-undersampling on dataset (creating artificial data that can bias the original dataset) or applying class weights for classification.

In the second part, we will examine the text corpus and perform a first objective analysis on the dataset by using LSA and the chi-square test.

Finally, we will test and compare different Machine Learning models and LSTMs and Bi-GRUs to see how they perform on a text classification task over the dataset created.
