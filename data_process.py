# %% [markdown]
# # DD2417 Final Project - Dating Historical Texts

# %% [markdown]
# ## Libraries + Imports

# %%
import os
import csv
import random
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seed all experiments and setup
random.seed(42)

# %% [markdown]
# ## Data - Setup and Analysis

# %% [markdown]
# ### Path Setup

# %%
# paths
raw_dataset_path = "./Datasets/raw_data"

# raw split
raw_train_split_path = "./Datasets/raw_train_split"
raw_test_split_path = "./Datasets/raw_test_split"

# clean split
clean_train_split_path = "./Datasets/clean_train_split"
clean_test_split_path = "./Datasets/clean_test_split"

# model
model_dataset_path = "./Datasets/model_data"


# %%
def cleanup(train_path, test_path):
    print(f"clean up train path - {train_path}")
    train_dir = os.listdir(train_path)
    train_dir.sort()
    for dir in train_dir:
        decade_path = os.path.join(train_path, dir)
        if os.path.isdir(decade_path):
            text_files = os.listdir(decade_path)
            text_files.sort()
            for file in text_files:
                if file.endswith(".txt"):
                    file_path = os.path.join(decade_path, file)
                    os.remove(file_path)
                    print(f"succesfully remove {file}")
            os.rmdir(decade_path)
            print(f"succesfully removed directory {dir}")
            print()

    print(f"clean up test path - {test_path}")
    test_dir = os.listdir(test_path)
    test_dir.sort()
    for dir in test_dir:
        decade_path = os.path.join(test_path, dir)
        if os.path.isdir(decade_path):
            text_files = os.listdir(decade_path)
            text_files.sort()
            for file in text_files:
                if file.endswith(".txt"):
                    file_path = os.path.join(decade_path, file)
                    os.remove(file_path)
                    print(f"succesfully remove {file}")
            os.rmdir(decade_path)
            print(f"succesfully removed directory {dir}")
            print()

    os.rmdir(train_path)
    print(f"succesfully removed {train_path}")
    os.rmdir(test_path)
    print(f"succesfully removed {test_path}")
    print(f"succesfully cleaned up training and test files")


# %%
def cleanup_model_data(model_dataset_path):
    print(f"clean up model path - {model_dataset_path}")
    files = os.listdir(model_dataset_path)

    # Remove each file
    for file in files:
        file_path = os.path.join(model_dataset_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"successfully removed {file}")
    os.rmdir(model_dataset_path)
    print(f"successfully removed {model_dataset_path}")


# %%
# raw split
if os.path.exists(raw_train_split_path) and os.path.exists(raw_test_split_path):
    cleanup(raw_train_split_path, raw_test_split_path)

if os.path.exists(model_dataset_path):
    cleanup_model_data(model_dataset_path)

os.makedirs(raw_train_split_path)
print(f"create raw train split directory")

os.makedirs(raw_test_split_path)
print(f"create raw test split directory")

os.makedirs(model_dataset_path)
print(f"create model data directory ")

# %%
# clean split
if os.path.exists(clean_train_split_path) and os.path.exists(clean_test_split_path):
    cleanup(clean_train_split_path, clean_test_split_path)

os.makedirs(clean_train_split_path)
print(f"create clean train split directory")
os.makedirs(clean_test_split_path)
print(f"create clean test split directory")

# %% [markdown]
# ### Book Data Analysis

# %%
# count all the data files in the raw data file
print(f"count the number of books in each decade directory in the raw data")
total_books = 0
for decade in range(1700, 1900, 10):
    decade_path = f"{raw_dataset_path}/{decade}"
    if os.path.exists(decade_path):
        text_files = [f for f in os.listdir(decade_path) if f.endswith(".txt")]
        print(f"{decade}: {len(text_files)} books")
        total_books += len(text_files)
print(f"total number of books for project: {total_books}")


# %%
# get all the titles of the books
def get_book_titles(dataset_path):
    book_titles = {}
    # filename_to_title = {}
    for year in range(1770, 1900, 10):
        decade_path = f"{dataset_path}/{year}"
        book_titles[year] = []

        # print(f"decade: {year}")
        text_files = sorted([f for f in os.listdir(decade_path) if f.endswith(".txt")])
        for filename in text_files:
            file_path = os.path.join(decade_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            title_match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
            book_title = title_match.group(1).strip()
            # print(f"book_title: {book_title}")
            # filename_to_title[filename] = book_title
            book_titles[year].append(book_title)
        # print(f"number of titles in decade: {year} -> {len(book_titles[year])}")
        print()
    return book_titles


book_titles = get_book_titles(raw_dataset_path)
print(f"{book_titles[1770]}")


# %%
def raw_dataset_info(dataset_path):
    years = [i for i in range(1770, 1900, 10)]
    book_titles = get_book_titles(dataset_path)

    book_data = []
    for decade in years:
        decade_path = f"{dataset_path}/{decade}"
        if os.path.exists(decade_path):
            text_files = sorted(
                [f for f in os.listdir(decade_path) if f.endswith(".txt")]
            )
            for index, filename in enumerate(text_files):
                if decade in book_titles and index < len(book_titles[decade]):
                    book_title = book_titles[decade][index]
                else:
                    book_title = f"unknown_book_{index + 1}"
                book_info = {
                    "decade": decade,
                    "filename": filename,
                    "book_title": book_title,
                    "filepath": os.path.join(decade_path, filename),
                    "book_id": f"{decade}_{book_title[:20].replace(' ', '_')}",
                }
                book_data.append(book_info)
    print(f"total number of books processed: {len(book_data)}")
    return book_data


raw_data_info = raw_dataset_info(raw_dataset_path)
print(f"The length of result after calling raw dataset info: {len(raw_data_info)}")

# %% [markdown]
# ## Data Split - Stratified Split of Books - Training Books, Testing Books


# %%
def create_stratified_split(book_data, train_split=0.8):
    train_books, test_books = [], []
    books_by_decade = {}

    books_by_decade = {}
    for book in book_data:
        decade = book["decade"]
        if decade not in books_by_decade:
            books_by_decade[decade] = []
        books_by_decade[decade].append(book)

    # debug check
    # for decade, books in books_by_decade.items():
    #     print(f"decade: {decade}, number of books: {len(books)}")

    for decade, books in sorted(books_by_decade.items()):
        shuffled_books = books.copy()
        random.shuffle(shuffled_books)

        total_books = len(books)
        train_size = max(1, int(total_books * train_split))
        test_size = total_books - train_size
        decade_train = shuffled_books[:train_size]
        decade_test = shuffled_books[train_size:]

        train_books.extend(decade_train)
        test_books.extend(decade_test)

    print(f"TRAIN BOOKS: {len(train_books)}")
    print(f"TEST BOOKS: {len(test_books)}")

    return train_books, test_books


raw_book_data = raw_dataset_info(raw_dataset_path)
raw_train, raw_test = create_stratified_split(raw_book_data)


# %%
def write_stratified_split(dataset, file_path):
    for i, book in enumerate(dataset):
        print(f"book: {i + 1}")
        # decade
        book_decade = str(book["decade"])
        # title
        book_title = book["book_title"]
        # filename
        book_filename = book["filename"]
        # path
        book_path = book["filepath"]

        print(f"read book <- {book_path}")
        with open(book_path, "r", encoding="utf-8") as f:
            raw_book = f.read()

        decade_path = os.path.join(file_path, book_decade)
        if not os.path.isdir(decade_path):
            os.makedirs(decade_path)
        out_file = decade_path + "/" + book_filename
        book["file_path"] = out_file
        print(f"new book filepath: {book_path}")
        print(f"write book -> {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(raw_book)
        print(f"wrote book successfully!!!")
        print()


# %%
write_stratified_split(raw_train, raw_train_split_path)

# %%
write_stratified_split(raw_test, raw_test_split_path)

# %% [markdown]
# ## Data-Preprocessing

# %% [markdown]
# ## Data-Cleaning


# %%
def clean_text(text):
    # remove everything up to and including start
    start_match = re.search(
        r"\*\*\* START OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL
    )
    if start_match:
        text = text[start_match.end() :]

    # remove everything after end
    end_match = re.search(r"\*\*\* END OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if end_match:
        text = text[: end_match.start()]

    # remove years
    text = re.sub(r"\b1[0-9]{3}\b", "", text)

    # remove whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# %%
def clean_stratified_split(raw_split_path, clean_split_path):
    decade_dirs = [
        dir
        for dir in os.listdir(raw_split_path)
        if os.path.isdir(os.path.join(raw_split_path, dir))
    ]
    decade_dirs.sort()

    total_books = 0
    for decade_dir in decade_dirs:
        clean_decade_path = os.path.join(clean_split_path, decade_dir)
        print(f"clean decade path: {clean_decade_path}")
        if not os.path.exists(clean_decade_path):
            os.makedirs(clean_decade_path)
        raw_decade_path = os.path.join(raw_split_path, decade_dir)
        print(f"raw decade path: {raw_decade_path}")
        text_files = [f for f in os.listdir(raw_decade_path) if f.endswith(".txt")]

        for text_file in text_files:
            total_books += 1
            print(f"books processed: {total_books}")
            raw_file_path = os.path.join(raw_decade_path, text_file)
            # print(f"raw data path: {raw_file_path}")
            clean_file_path = os.path.join(clean_decade_path, text_file)
            # print(f"clean file path: {clean_file_path}")
            print(
                f"read raw data: {raw_file_path} -> clean data -> write clean data: {clean_file_path}"
            )
            with open(raw_file_path, "r", encoding="utf-8") as f:
                raw_data = f.read()
                print(f"read raw data <- {raw_file_path}")

            cleaned_data = clean_text(raw_data)
            with open(clean_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_data)
                print(f"write clean data -> {clean_file_path}")

            print(f"wrote cleaned data successfully!!!")
            print()


# %%
clean_stratified_split(raw_train_split_path, clean_train_split_path)

# %%
clean_stratified_split(raw_test_split_path, clean_test_split_path)

# %% [markdown]
# ##


# %%
def create_paragraphs(text, min_words=10, max_words=210):
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


# %%
def create_paragraph_data(clean_split_path, raw_data_path):
    years = sorted(os.listdir(clean_split_path))
    print(f"years in sorted order: {years}")
    # print(f"dataset path: {os.listdir(clean_train_split_path)}")
    book_titles = get_book_titles(raw_data_path)
    paragraph_data = []
    total_books = 0

    # all the years
    for decade in years:
        print(f"process decade: {decade}")
        decade_path = f"{clean_split_path}/{decade}"
        print(f"decade_path: {decade_path}")
        decade_titles = book_titles[int(decade)]
        print(f"number of book titles in decade: {len(decade_titles)}")

        if os.path.exists(decade_path):
            text_files = sorted(
                [f for f in os.listdir(decade_path) if f.endswith(".txt")]
            )
            for index, text_filename in enumerate(text_files):
                print(f"current book: {index + 1}")
                total_books += 1
                print(f"book filename: {text_filename}")
                text_file_number = int(re.findall(r"\d+", text_filename)[0])
                print(f"book number: {text_file_number}")
                book_title = decade_titles[text_file_number - 1]
                text_filepath = os.path.join(decade_path, text_filename)
                with open(text_filepath, "r", encoding="utf-8") as f:
                    clean_book_data = f.read()
                    print(f"succesfully read book!!!")
                book_paragraphs = create_paragraphs(clean_book_data)
                print(f"number of paragraphs created: {len(book_paragraphs)}")
                paragraph_length = len(book_paragraphs[0].split())
                print(f"length of a paragraph: {paragraph_length}")

                paragraph_info = {
                    "paragraphs": book_paragraphs,
                    "book_title": book_title,
                    "decade": decade,
                    "filepath": text_filepath,
                    "book_id": f"{decade}_{book_title[:20].replace(' ', '_')}",
                }
                paragraph_data.append(paragraph_info)

        print(f"total number of books processed in decade {decade} -> {total_books}")
        print()

    print(f"total number of books processed: {total_books}")
    return paragraph_data


# %%
paragraph_train_data = create_paragraph_data(clean_train_split_path, raw_dataset_path)

# %%
paragraph_test_data = create_paragraph_data(clean_test_split_path, raw_dataset_path)


# %%
def write_data_to_csv(paragraph_data, output_file):
    print(f"write {len(paragraph_data)} -> {output_file}")

    # map decade to label for classification
    decades = sorted(set(int(item["decade"]) for item in paragraph_data))
    decade_to_label = {decade: idx for idx, decade in enumerate(decades)}

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        header_fields = [
            "text",
            "decade",
            "decade_label",
            "book_title",
            "book_id",
            "paragraph_id",
            "word_count",
        ]
        writer = csv.DictWriter(f, fieldnames=header_fields)

        writer.writeheader()
        total_paragraphs = 0
        total_book_count = 0

        for book_index, paragraph_info in enumerate(paragraph_data):
            decade = int(paragraph_info["decade"])
            decade_label = decade_to_label[decade]
            book_title = paragraph_info["book_title"]
            book_id = paragraph_info["book_id"]
            paragraphs = paragraph_info["paragraphs"]

            for index, text in enumerate(paragraphs):
                clean_text = text.replace("\n", " ").replace("\r", " ").strip()
                word_count = len(clean_text.split())

                # from the paper
                if word_count < 10:
                    continue

                writer.writerow(
                    {
                        "text": clean_text,
                        "decade": decade,
                        "decade_label": decade_label,
                        "book_title": book_title,
                        "book_id": book_id,
                        "paragraph_id": f"{decade}_{book_id}_{index:03d}",
                        "word_count": word_count,
                    }
                )
                total_paragraphs += 1

            total_book_count += 1
            print(f"processed total books: {book_index + 1}")

    print(
        f"Succesfully wrote {total_paragraphs} paragraphs and processed {total_book_count} books -> {output_file}"
    )


# %%
out_file = os.path.join(model_dataset_path, "train_data.csv")
write_data_to_csv(paragraph_train_data, out_file)

# %%
out_file = os.path.join(model_dataset_path, "test_data.csv")
write_data_to_csv(paragraph_test_data, out_file)

# %%
# cleanup_model_data(model_dataset_path)
