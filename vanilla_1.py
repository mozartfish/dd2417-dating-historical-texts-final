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
# ### Tokenizer

# %%


# %% [markdown]
# ### Train Data - First paragraph

# %%
train_data_first = train_df.iloc[0, 0]
train_data_first

# %% [markdown]
# ### Test Data - First Paragraph

# %%
test_data_first = test_df.iloc[0, 0]
test_data_first
