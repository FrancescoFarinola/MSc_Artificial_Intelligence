"""
Load data from disk
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from settings import DATA_DIR


def load_dataset(data_dir=DATA_DIR, filename="training_set.json"):
    """
    Load the dataset
    @param data_dir: data directory
    @param filename: dataset JSON file
    @return: dataframe
    """

    # open file to read
    with open(f"{data_dir}/{filename}") as f:
        data = json.load(f)

    # read file
    dataframe_rows = []
    for d in data["data"]:
        title = d["title"]
        paragraphs = d["paragraphs"]
        for p in paragraphs:
            context = p["context"]
            qas = p["qas"]
            for q in qas:
                answers = q["answers"]
                question = q["question"]
                qid = q["id"]
                for a in answers:
                    answer_start = a["answer_start"]
                    text = a["text"]

                    # single row of the dataframe
                    dataframe_row = {
                        "title": title,
                        "context": context,
                        "answer_start": answer_start,
                        "text": text,
                        "question": question,
                        "id": qid
                    }
                    dataframe_rows.append(dataframe_row)

    return pd.DataFrame(dataframe_rows)


def load_dataset_without_answer(path):
    """
    Load dataset without answers
    @param path: dataset file path
    @return: dataset
    """
    # open file to read
    with open(path) as f:
        data = json.load(f)

    # read file
    dataframe_rows = []
    for d in data["data"]:
        title = d["title"]
        paragraphs = d["paragraphs"]
        for p in paragraphs:
            context = p["context"]
            qas = p["qas"]
            for q in qas:
                question = q["question"]
                qid = q["id"]

                # dataframe row
                dataframe_row = {
                    "title": title,
                    "context": context,
                    "question": question,
                    "id": qid
                }
                dataframe_rows.append(dataframe_row)

    return pd.DataFrame(dataframe_rows)


def remove_rows(dataframe):
    """
    Remove rows containing errors or more instances of the proposed answer
    @param dataframe: dataset
    @return: test dataset
    """
    # dataset with multiple occurrences of the answer
    occurrences = dataframe.apply(lambda x: x.context.count(x.text), axis=1)
    idx_multiple_occurrences = np.where(occurrences > 1)
    ts_df1 = dataframe.loc[idx_multiple_occurrences]
    ts_df1.reset_index(inplace=True, drop=True)

    ts_df2 = dataframe.loc[~ dataframe.id.isin(ts_df1.id)]
    ts_df2.reset_index(inplace=True, drop=True)
    # dataset with errors
    idx_errors = ts_df2.apply(lambda row: row.answer_start != row.context.find(row.text), axis=1)
    ts_df2 = ts_df2.loc[idx_errors]
    ts_df2.reset_index(inplace=True, drop=True)

    # return the concatenation of the two datasets
    ts_df = pd.concat([ts_df1, ts_df2])
    ts_df.reset_index(inplace=True, drop=True)
    return ts_df


def split_test_set(dataframe):
    """
    Split training and test set
    @param dataframe: dataset
    @return: training dataset, test dataset
    """
    # remove rows containing errors or containing multiple instances of the answer
    # keep removed rows as test set
    ts_df = remove_rows(dataframe)

    # reset indices
    dataframe = dataframe[~dataframe['id'].isin(ts_df.id)]
    dataframe.reset_index(inplace=True, drop=True)

    # return training set, test set
    return dataframe, ts_df


def split_validation_set(dataframe, rate):
    """
    Split dataframe into training and validation set
    nb: records with the same title are kept together
    @param dataframe: dataframe to split
    @param rate: validation ratio
    @return: training set, validation set
    """
    # split dataframe
    tr_title, val_title = train_test_split(np.unique(dataframe.title), test_size=rate, random_state=0)
    tr_idx = np.isin(dataframe.title, tr_title)
    val_idx = np.isin(dataframe.title, val_title)

    # reset indices
    tr_df = dataframe.loc[tr_idx]
    tr_df.reset_index(inplace=True, drop=True)
    val_df = dataframe.loc[val_idx]
    val_df.reset_index(inplace=True, drop=True)

    # return training set, validation set
    return tr_df, val_df
