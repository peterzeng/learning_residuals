# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from argparse import ArgumentParser

import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm

random.seed(43)
nlp = English()
nlp.add_pipe('sentencizer')


def gather_author_data(data, truth):
    """Returns a dictionary where each key is the author identifier and
       each value is a list of tuples (author_id, fandom, document).
    """
    assert len(data) == len(truth)
    N = len(data)

    author_data = defaultdict(list)
    seen_data = defaultdict(set)

    for idx in range(N):
        truth_row = truth.iloc[idx]

        author = truth_row.authors[0]
        sample = (author, data.iloc[idx].fandoms[0], data.iloc[idx].pair[0])

        if not sample[-1] in seen_data[author]:
            author_data[author].append(sample)
            seen_data[author].add(data.iloc[idx].pair[1])

        author = truth_row.authors[1]
        sample = (author, data.iloc[idx].fandoms[1], data.iloc[idx].pair[1])

        if not sample[-1] in seen_data[author]:
            author_data[author].append(sample)
            seen_data[author].add(data.iloc[idx].pair[1])

    return author_data


def paragraph_split(syms, topic):
    """Split story into paragraphs.
    """
    doc = nlp(syms)
    sentences = [str(s) for s in doc.sents]

    new_data = {
        "syms": [],
        "topic": [],
    }

    N = len(sentences)
    paragraph_length = 8

    for i in range(0, N, paragraph_length):
        paragraph = ' '.join(sentences[i:i + 8])
        new_data["syms"].append(paragraph)
        new_data["topic"].append(topic)

    return new_data


def process_author(data):
    """Takes an author's data, breaks it down into paragraph, and
       formats it correctly.

    Args:
        data (list): List of tuples (author_id, fandom, document).

    Returns:
        dict: Dictionary with the symbols, topic, and whether or not it
              belongs to the dev set.
    """
    new_data = {
        "syms": [],
        "topic": []
    }

    for d in data:
        split_data = paragraph_split(syms=d[-1], topic=d[1])
        new_data["syms"].extend(split_data["syms"])
        new_data["topic"].extend(split_data["topic"])

    return new_data


def get_results(data_path, truth_path):
    data = pd.read_json(data_path, lines=True)
    truth = pd.read_json(truth_path, lines=True)

    author_data = gather_author_data(data, truth)

    print("Gathering author data")
    pool = Pool(40)
    results = list(tqdm(pool.imap(process_author, author_data.values()), total=len(author_data)))
    pool.close()

    authors = list(author_data.keys())
    author_ids = dict(zip(authors, range(len(authors))))

    for author, r in zip(author_data.keys(), results):
        r["author_id"] = author_ids[author]
    return results


def save_to_json(results, out_path):
    with open(out_path, 'w+') as f:
        for line in results:
            f.write(json.dumps(line))
            f.write('\n')


def main():
    train_results = get_results('train_data.jsonl', 'train_truth.jsonl')
    test_results = get_results('test_data.jsonl', 'test_truth.jsonl')
    split = len(train_results) // 6

    save_to_json(train_results[split:], 'train_processed.jsonl')
    save_to_json(train_results[:split], 'dev_processed.jsonl')
    save_to_json(test_results, 'test_processed.jsonl')

    return 0


if __name__ == "__main__":
    sys.exit(main())