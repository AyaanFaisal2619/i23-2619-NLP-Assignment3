import os
import re
import string
from collections import Counter

import pandas as pd
import torch

DATASET_DIR = "Dataset"
LEGACY_DATASET_DIR = "."
TRAIN_FILE = "dataset_train.csv"
VAL_FILE = "dataset_val.csv"
TEST_FILE = "dataset_test.csv"
TEXT_COL = "reviewText"
TOKENS_COL = "tokens"
OUTPUT_DIR = "Dataset"
OUTPUT_FILE = "dataset.pt"
MAX_LEN = 100
MIN_FREQ = 2
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_data(path):
    return pd.read_csv(path)


def resolve_dataset_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), DATASET_DIR, filename),
        os.path.join(script_dir, DATASET_DIR, filename),
        os.path.join(os.path.dirname(script_dir), DATASET_DIR, filename),
        os.path.join(os.getcwd(), LEGACY_DATASET_DIR, filename),
        os.path.join(script_dir, LEGACY_DATASET_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in Dataset/ (cwd/script/parent) or project root."
    )


def clean_text(df):
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower()
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: re.sub(r"\d+", "", x))
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    return df


def tokenize(df):
    df = df.copy()
    df[TOKENS_COL] = df[TEXT_COL].apply(lambda x: x.split())
    return df


def build_vocab(token_lists, min_freq=MIN_FREQ):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]


def pad(sequence, max_len=MAX_LEN):
    if len(sequence) > max_len:
        return sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))


def process(df, vocab):
    input_ids = df[TOKENS_COL].apply(lambda tokens: pad(encode(tokens, vocab)))
    return torch.tensor(input_ids.tolist(), dtype=torch.long)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = tokenize(clean_text(load_data(resolve_dataset_path(TRAIN_FILE))))
    val_df = tokenize(clean_text(load_data(resolve_dataset_path(VAL_FILE))))
    test_df = tokenize(clean_text(load_data(resolve_dataset_path(TEST_FILE))))

    vocab = build_vocab(train_df[TOKENS_COL])

    train_tensor = process(train_df, vocab)
    val_tensor = process(val_df, vocab)
    test_tensor = process(test_df, vocab)

    print(train_df[[TEXT_COL]].head())
    print("Train tensor shape:", train_tensor.shape)
    print("Vocab size:", len(vocab))

    payload = {"train": train_tensor, "val": val_tensor, "test": test_tensor, "vocab": vocab}
    torch.save(payload, os.path.join(OUTPUT_DIR, OUTPUT_FILE))
    print("Processed data saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()