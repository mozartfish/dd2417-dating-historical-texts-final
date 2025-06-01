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

    def get_vocab_size(self):
        return len(self.word2id)


# %% [markdown]
# ### Tokenize Data

# %%
text_tokenizer = HistoricalTextTokenizer()

# %% [markdown]
# #### Training Data

# %%
train_text_data, train_labels = text_tokenizer.process_files(clean_train_split_path)

# %% [markdown]
# #### Testing Data

# %%
test_text_data, test_labels = text_tokenizer.process_files(clean_test_split_path)

# %% [markdown]
# ### Create Labels

# %%
labels = sorted(set(train_labels + test_labels))
decade2label = {decade: i for i, decade in enumerate(labels)}
print(f"{decade2label}")

# %% [markdown]
# ### Check Tokenizer + Labels

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
train_sample = train_text_data[0]
train_sample_label = train_labels[0]
word_ids = text_tokenizer.tokenize_text_to_id(train_sample)

print(f"train sample -> {train_sample}")
print(f"train sample labe -> {train_sample_label}")
print(f"tokenized train_sample -> {word_ids}")
print(f"length of tokenized word {len(word_ids)}")

# %%
test_sample = test_text_data[0]
test_sample_label = test_labels[0]
word_ids = text_tokenizer.tokenize_text_to_id(test_sample)

print(f"test sample -> {test_sample}")
print(f"test sample label -> {test_sample_label}")
print(f"tokenized test_sample -> {word_ids}")
print(f"length of tokenized word {len(word_ids)}")

# %% [markdown]
# ## Vanilla RNN
# This code structure is adapted from Professor Johan Boye's NER PyTorch Code

# %%
import csv
from tqdm import tqdm
import string
import codecs
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# %% [markdown]
# ## Tokenize Data - Training, Testing

# %%
# Run this cell to init mappings from characters to IDs and back again,
# from words to IDs and back again, and from labels to IDs and back again

# create new tokenizer object
print(f"create new tokenizer")
tokenizer = HistoricalTextTokenizer()

print(f"tokenize -> {clean_train_split_path}")
train_data, train_labels = tokenizer.process_files(clean_train_split_path)
print(f"succesfully tokenized <- {clean_train_split_path}")

print(f"tokenize -> {clean_test_split_path}")
test_data, test_labels = tokenizer.process_files(clean_test_split_path)
print(f"succesfully tokenized <- {clean_test_split_path}")

print(f"create decade labels")
decades = sorted(set(train_labels + test_labels))
decade_to_label = {decade: i for i, decade in enumerate(decades)}
label_to_decade = {i: decade for i, decade in enumerate(decades)}
print(f"successfully created decades labels")

UNKNOWN = "<unk>"  # Unknown char or unknown word
PADDING_WORD = "<pad>"
id_to_label = [f"decade_{decade}" for decade in decades]


def label_to_id(decade):
    return decade_to_label[decade]


# %% [markdown]
# ## Glove Embeddings Setup


# %%
def load_glove_embeddings(
    embedding_file, padding_word=PADDING_WORD, unknown_word=UNKNOWN
):
    """
    Reads Glove embeddings from a file.

    Returns vector dimensionality, the word_to_id mapping (as a dict),
    and the embeddings (as a list of lists).
    """
    word_to_id = {}  # Dictionary to store word-to-ID mapping
    word_to_id[padding_word] = 0
    word_to_id[unknown_word] = 1
    embeddings = []
    with open(embedding_file, encoding="utf8") as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word_to_id[word] = len(word_to_id)
    D = len(embeddings[0])

    embeddings.insert(
        word_to_id[padding_word], [0] * D
    )  # <PAD> has an embedding of just zeros
    embeddings.insert(
        word_to_id[unknown_word], [-1] * D
    )  # <UNK> has an embedding of just minus-ones

    return D, word_to_id, embeddings


# %% [markdown]
# ## DataLoaders - Convert text for RNN


# %%
class HistoricalTextDataset(Dataset):
    """
    A class loading NER dataset from a CSV file to be used as an input
    to PyTorch DataLoader.

    The CSV file has 4 fields: sentence number (only at the start of a new
    sentence), word, POS tag (ignored), and label.

    Datapoints are sentences + associated labels for each word. If the
    words have not been seen before (i.e, they are not found in the
    'word_to_id' dict), they will be mapped to the unknown word '<UNK>'.
    """

    def __init__(self, texts, labels, word_to_id, decade_to_label):
        self.texts = texts
        self.labels = labels
        self.word_to_id = word_to_id
        self.decade_to_label = decade_to_label

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        decade = self.labels[idx]
        label_id = self.decade_to_label[decade]

        return text, label_id


# %%
# Let's check out some of these data structures
dim, word_to_id, embeddings = load_glove_embeddings(glove_6b_50_path)
tokenizer.word2id = word_to_id
tokenizer.id2word = {v: k for k, v in word_to_id.items()}

print("The embedding for the word 'good' looks like this:")
print(embeddings[word_to_id["good"]])
print()

# Read the data we are going to use for testing the model
test_set = HistoricalTextDataset(test_data, test_labels, word_to_id, decade_to_label)
print("There are", len(test_set), "documents in the testset")
dp = 0
text, label = test_set[dp]
print("Document", dp, "starts with:", text[:100], "...")
print("It has the label", label, "which corresponds to decade", label_to_decade[label])

# %% [markdown]
# ## Padding Sequences


# %%
def pad_sequence_documents(batch, padding_word=PADDING_WORD, max_length=500):
    batch_data, batch_labels = zip(*batch)

    # Convert documents to word IDs
    padded_data = []
    for text in batch_data:
        word_ids = tokenizer.tokenize_text_to_id(text)

        # Truncate if too long (shouldn't happen with max_length=500)
        if len(word_ids) > max_length:
            word_ids = word_ids[:max_length]

        # Pad if too short
        padding_id = word_to_id[padding_word]
        while len(word_ids) < max_length:
            word_ids.append(padding_id)

        padded_data.append(word_ids)

    padded_labels = list(batch_labels)
    return padded_data, padded_labels


# %%
x = [(train_data[0], train_labels[0])]
pad_sequence_documents(x)

# %% [markdown]
# ## RNN Architecture


# %%
class DocumentClassifier(nn.Module):
    def __init__(
        self,
        word_embeddings,  # Pre-trained word embeddings
        word_to_id,  # Mapping from words to ids
        num_classes=13,  # Number of decades to classify
        word_hidden_size=128,  # Hidden size of the RNN (paper uses 128)
        padding_word=PADDING_WORD,
        unknown_word=UNKNOWN,
        device="cpu",
    ):
        super(DocumentClassifier, self).__init__()
        self.padding_word = padding_word
        self.unknown_word = unknown_word
        self.word_to_id = word_to_id
        self.word_hidden_size = word_hidden_size
        self.device = device
        self.num_classes = num_classes

        # Create an embedding tensor for the words and import the Glove
        # embeddings. The embeddings are frozen (i.e., they will not be
        # updated during training).
        vocabulary_size = len(word_embeddings)
        self.word_emb_size = len(word_embeddings[0])

        self.word_emb = nn.Embedding(vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(
            torch.tensor(word_embeddings, dtype=torch.float), requires_grad=False
        )

        self.rnn = nn.RNN(
            input_size=self.word_emb_size,
            hidden_size=word_hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )

        # Document Classification
        self.final_pred = nn.Linear(word_hidden_size, num_classes)

    def forward(self, x):
        # Shape: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        word_embeddings = self.word_emb(x)
        h0 = torch.zeros(1, batch_size, self.word_hidden_size).to(x.device)
        rnn_output, hidden = self.rnn(word_embeddings, h0)
        final_hidden = hidden.squeeze(0)
        logits = self.final_pred(final_hidden)
        return logits


# %% [markdown]
# ## Train and Evaluate the Model - Small Data

# %%
# # ================== Hyper-parameters ==================== #

learning_rate = 0.0001
epochs = 5  # paper uses 5 epochs
# ======================= Training (First 50 documents) ======================= #

if torch.backends.mps.is_available():
    device = "mps"
    print("Running on MGPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Running on CUDA")
else:
    device = "cpu"
    print("Running on CPU")

dim, word_to_id, embeddings = load_glove_embeddings(glove_6b_50_path)
tokenizer.word2id = word_to_id
tokenizer.id2word = {v: k for k, v in word_to_id.items()}

# Use only first 50 documents for testing
print("Using first 10000 documents for testing...")
train_data_small = train_data[:10000]
train_labels_small = train_labels[:10000]
test_data_small = test_data[:50]  # First 20 for testing
test_labels_small = test_labels[:50]

training_set = HistoricalTextDataset(
    train_data_small, train_labels_small, word_to_id, decade_to_label
)
test_set = HistoricalTextDataset(
    test_data_small, test_labels_small, word_to_id, decade_to_label
)

training_loader = DataLoader(
    training_set, batch_size=16, collate_fn=pad_sequence_documents
)  # Smaller batch size
test_loader = DataLoader(test_set, batch_size=16, collate_fn=pad_sequence_documents)

print(f"Training on {len(training_set)} documents")
print(f"Testing on {len(test_set)} documents")

rnn_classifier = DocumentClassifier(
    word_embeddings=embeddings,
    word_to_id=word_to_id,
    num_classes=len(decade_to_label),
    word_hidden_size=128,
    device=device,
).to(device)

optimizer = optim.Adam(rnn_classifier.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.CrossEntropyLoss()

rnn_classifier.train()
for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
        x = torch.tensor(x, dtype=torch.long).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = rnn_classifier(x)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(rnn_classifier.parameters(), 5)
        optimizer.step()

        # Track training accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    scheduler.step()

    train_acc = 100 * correct / total
    avg_loss = epoch_loss / len(training_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%"
    )

# %%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

rnn_classifier.eval()

# Create test dataset and loader
test_set = HistoricalTextDataset(test_data, test_labels, word_to_id, decade_to_label)
test_loader = DataLoader(test_set, batch_size=16, collate_fn=pad_sequence_documents)

all_predictions = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        pred = torch.argmax(rnn_classifier(x), dim=-1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        all_predictions.extend(pred)
        all_labels.extend(y_np)

# confusion matrix
num_classes = len(decade_to_label)
confusion_matrix_manual = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for i in range(len(all_predictions)):
    actual = all_labels[i]
    predicted = all_predictions[i]
    confusion_matrix_manual[actual][predicted] += 1

# Print results
print("Confusion Matrix:")
print("Predicted ->", [f"D{label_to_decade[i]}" for i in range(num_classes)])
for i, row in enumerate(confusion_matrix_manual):
    print(f"Actual D{label_to_decade[i]}: {row}")

accuracy = (
    confusion_matrix_manual[0][0]
    + sum(confusion_matrix_manual[i][i] for i in range(num_classes))
) / sum(sum(row) for row in confusion_matrix_manual)
print(f"Accuracy: {accuracy:.4f}")

# %% [markdown]
# ## Train and Evaluate the Model - All Data

# %%
# ================== Hyper-parameters ==================== #
learning_rate = 0.0001
epochs = 5

# ======================= Training ======================= #
if torch.backends.mps.is_available():
    device = "mps"
    print("Running on MGPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Running on CUDA")
else:
    device = "cpu"
    print("Running on CPU")

dim, word_to_id, embeddings = load_glove_embeddings(glove_6b_50_path)
tokenizer.word2id = word_to_id
tokenizer.id2word = {v: k for k, v in word_to_id.items()}

training_set = HistoricalTextDataset(
    train_data, train_labels, word_to_id, decade_to_label
)
training_loader = DataLoader(
    training_set, batch_size=64, collate_fn=pad_sequence_documents
)

rnn_classifier = DocumentClassifier(
    word_embeddings=embeddings,
    word_to_id=word_to_id,
    num_classes=len(decade_to_label),
    word_hidden_size=128,
    device=device,
).to(device)

optimizer = optim.Adam(rnn_classifier.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.CrossEntropyLoss()

rnn_classifier.train()
for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
        x = torch.tensor(x, dtype=torch.long).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = rnn_classifier(x)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(rnn_classifier.parameters(), 5)
        optimizer.step()

        # Track training accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    scheduler.step()

    train_acc = 100 * correct / total
    avg_loss = epoch_loss / len(training_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%"
    )

# %%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

rnn_classifier.eval()

# Create test dataset and loader
test_set = HistoricalTextDataset(test_data, test_labels, word_to_id, decade_to_label)
test_loader = DataLoader(test_set, batch_size=16, collate_fn=pad_sequence_documents)

all_predictions = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        pred = torch.argmax(rnn_classifier(x), dim=-1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        all_predictions.extend(pred)
        all_labels.extend(y_np)

# confusion matrix
num_classes = len(decade_to_label)
confusion_matrix_manual = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for i in range(len(all_predictions)):
    actual = all_labels[i]
    predicted = all_predictions[i]
    confusion_matrix_manual[actual][predicted] += 1

# Print results
print("Confusion Matrix:")
print("Predicted ->", [f"D{label_to_decade[i]}" for i in range(num_classes)])
for i, row in enumerate(confusion_matrix_manual):
    print(f"Actual D{label_to_decade[i]}: {row}")

accuracy = (
    confusion_matrix_manual[0][0]
    + sum(confusion_matrix_manual[i][i] for i in range(num_classes))
) / sum(sum(row) for row in confusion_matrix_manual)
print(f"Accuracy: {accuracy:.4f}")
