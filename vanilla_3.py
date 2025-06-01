# %% [markdown]
# # RNN Architecture - Vanilla RNN

# %% [markdown]
# ## Setup - Libraries, Packages, Embeddings, Paths

# %% [markdown]
# ### Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
from tqdm import tqdm

# %% [markdown]
# ### Glove Embeddings

# %%
embeddings_path = "./Embeddings"


def download_progress(block_num, block_size, total_size):
    if not hasattr(download_progress, "pbar"):
        download_progress.pbar = tqdm(total=total_size, unit="B", unit_scale=True)
    download_progress.pbar.update(block_size)


if not os.path.exists(embeddings_path):
    print(f"create directory to store pre-trained glove embeddings")
    os.makedirs(embeddings_path)
    print(f"download pre-trained Glove Embeddings")
    urllib.request.urlretrieve(
        "http://nlp.stanford.edu/data/glove.6B.zip",
        "./Embeddings/glove.6B.zip",
        download_progress,
    )
    print("unpack embeddings")
    with zipfile.ZipFile("./Embeddings/glove.6B.zip", "r") as zip_ref:
        zip_ref.extractall("./Embeddings/")
    os.remove("./Embeddings/glove.6B.zip")

    print("embeddings download complete")

# %% [markdown]
# ### Paths

# %%
glove_6b_50_path = "./Embeddings/glove.6B.50d.txt"
train_data_path = "./Datasets/model_data/train_data.csv"
test_data_path = "./Datasets/model_data/test_data.csv"
clean_train_split_path = "./Datasets/clean_train_split/"
clean_test_split_path = "./Datasets/clean_test_split"

# %% [markdown]
# ## Data

# %% [markdown]
# ### Train Data

# %%
train_df = pd.read_csv(train_data_path)
train_df.head(10)

# %%
train_df.info()

# %% [markdown]
# ### Test Data

# %%
test_df = pd.read_csv(test_data_path)
test_df.head(10)

# %%
test_df.info()

# %% [markdown]
# ## Tokenization

# %% [markdown]
# ### NLTK Tokenizer
# This code was adapted from Professor Johan Boye's DD2417 assignment tokenizers

# %%
import nltk

nltk.download("punkt_tab")
import numpy as np
from collections import defaultdict


class HistoricalTextTokenizer:
    """
    All of this code is adapted from Professor Johan Boye's DD2417 assignment tokenizers
    """

    def __init__(self):
        self.word2id = defaultdict(lambda: None)
        self.id2word = defaultdict(lambda: None)
        self.latest_new_word = -1
        self.tokens_processed = 0

        self.UNKNOWN = "<unk>"
        self.PADDING_WORD = "<pad>"

        self.get_word_id(self.PADDING_WORD)
        self.get_word_id(self.UNKNOWN)

    def get_word_id(self, word):
        word = word.lower()
        if word in self.word2id:
            return self.word2id[word]
        else:
            self.latest_new_word += 1
            self.id2word[self.latest_new_word] = word
            self.word2id[word] = self.latest_new_word
            return self.latest_new_word

    def process_files(self, file_or_dir):
        all_texts = []
        all_labels = []

        if os.path.isdir(file_or_dir):
            decade_dirs = sorted(
                [
                    d
                    for d in os.listdir(file_or_dir)
                    if os.path.isdir(os.path.join(file_or_dir, d))
                ]
            )
            for decade_dir in decade_dirs:
                decade_path = os.path.join(file_or_dir, decade_dir)
                decade = int(decade_dir)
                print(f"Processing decade: {decade}")
                text_files = sorted(
                    [f for f in os.listdir(decade_path) if f.endswith(".txt")]
                )
                print(f"number of files in {decade} directory: {len(text_files)}")

                for file in text_files:
                    filepath = os.path.join(decade_path, file)
                    print(f"tokenize file {file}")
                    text, labels = self.process_file(filepath, decade)
                    all_texts.extend(text)
                    all_labels.extend(labels)
        else:
            texts, labels = self.process_file(file_or_dir, 0)
            all_texts.extend(texts)
            all_labels.extend(labels)

        return all_texts, all_labels
        # pass

    def process_file(self, filepath, decade):
        print(filepath)
        stream = open(filepath, mode="r", encoding="utf-8", errors="ignore")
        text = stream.read()
        stream.close()

        try:
            self.tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            self.tokens = nltk.word_tokenize(text)

        for i, token in enumerate(self.tokens):
            self.tokens_processed += 1
            word_id = self.get_word_id(token)

            if self.tokens_processed % 10000 == 0:
                print("Processed", "{:,}".format(self.tokens_processed), "tokens")

        paragraphs = self.create_paragraphs(text)
        labels = [decade] * len(paragraphs)

        return paragraphs, labels
        # pass

    def create_paragraphs(self, text, min_words=10, max_words=210):
        words = text.split()
        paragraphs = []
        start = 0

        while start < len(words):
            end = min(start + max_words, len(words))
            paragraph_words = words[start:end]
            if len(paragraph_words) >= min_words:
                paragraph_text = " ".join(paragraph_words)
                paragraphs.append(paragraph_text)
            start = end

        return paragraphs
        # pass

    def tokenize_text_to_id(self, text):
        try:
            tokens = nltk.word_tokenize(text.lower())
        except LookupError:
            nltk.download("punkt")
            tokens = nltk.word_tokenize(text.lower())
        word_ids = []
        for token in tokens:
            if token in self.word2id:
                word_ids.append(self.word2id[token])
            else:
                word_ids.append(self.word2id[self.UNKNOWN])
        return word_ids

        # pass

    def pad_sequence_to_length(self, word_ids, max_length=220):
        padding_id = self.word2id[self.PADDING_WORD]
        if len(word_ids) > max_length:
            word_ids = word_ids[:max_length]

        while len(word_ids) < max_length:
            word_ids.append(padding_id)
        return word_ids
        # pass

    def get_vocab_size(self):
        return len(self.word2id)
        # pass


# %% [markdown]
# ### Tokenize Train Data

# %%
book_tokenizer = HistoricalTextTokenizer()
train_text_data, train_labels = book_tokenizer.process_files(clean_train_split_path)

# %% [markdown]
# ### Tokenize Test Data

# %%
book_tokenizer = HistoricalTextTokenizer()
test_text_data, test_labels = book_tokenizer.process_files(clean_test_split_path)

# %% [markdown]
# ### Create Labels

# %%
print(f"number of train labels -> {len(train_labels)}")
print(f"length of train text(paragraphs) -> {len(train_text_data)}")
print()

print(f"number of test labels -> {len(test_labels)}")
print(f"length of test text -> {len(test_text_data)}")
print()

print(f"train text {train_text_data[0]}")
print(f"train label {train_labels[0]}")
print()

print(f"test text(paragraphs) {test_text_data[0]}")
print(f"test label {test_labels[0]}")

# %%
labels = sorted(set(train_labels + test_labels))
decade2label = {decade: i for i, decade in enumerate(labels)}
print(f"{decade2label}")

# %% [markdown]
# ### Tokenize Test - Train Data

# %%
train_sample = train_text_data[0]
train_sample_label = train_labels[0]
word_ids = book_tokenizer.tokenize_text_to_id(train_sample)

print(f"train sample -> {train_sample}")
print(f"train sample labe -> {train_sample_label}")
print(f"tokenized train_sample -> {word_ids}")
print(f"length of tokenized word {len(word_ids)}")

# %% [markdown]
# ### Tokenize Test - Test Data

# %%
test_sample = test_text_data[0]
test_sample_label = test_labels[0]
word_ids = book_tokenizer.tokenize_text_to_id(test_sample)

print(f"test sample -> {test_sample}")
print(f"test sample label -> {test_sample_label}")
print(f"tokenized test_sample -> {word_ids}")
print(f"length of tokenized word {len(word_ids)}")

# %%
