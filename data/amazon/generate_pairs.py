# Script to generate pairs from amazon data and save it to a CSV file
# Usage: generate_pairs <processed.jsonl file> <output_file>

import json
import random
import sys

import pandas as pd

post_to_id = {}
post_id = 0


def get_post_id(doc):
    global post_id
    if doc in post_to_id:
        return post_to_id[doc]

    post_to_id[doc] = post_id
    post_id += 1
    return post_to_id[doc]


def generate_pairs(data_path, out_path, num_pairs):
    data_list = []
    same_pairs = []
    diff_pairs = []

    with open(data_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data_list.append((json_data['syms'], json_data['topic'], json_data['author_id']))

    for _ in range(num_pairs // 2):
        # Sample same pair
        docs = topics = author_id = None

        while True:
            docs, topics, author_id = random.choice(data_list)
            if len(docs) >= 2:
                break

        i, j = random.sample(range(len(docs)), 2)
        author_id = str(author_id)

        same_pairs.append({
            'post_1': docs[i], 'post_2': docs[j],
            'post_1_id': get_post_id(docs[i]), 'post_2_id': get_post_id(docs[j]),
            'author_1': author_id, 'author_2': author_id,
            'topic_1': topics[i], 'topic_2': topics[j], 'same': True
        })

        # Sample different pair
        author_docs1, author_docs2 = random.sample(data_list, 2)
        docs1, topics1, author1 = author_docs1
        docs2, topics2, author2 = author_docs2

        i = random.choice(range(len(docs1)))
        j = random.choice(range(len(docs2)))

        diff_pairs.append({
            'post_1': docs1[i], 'post_2': docs2[j],
            'post_1_id': get_post_id(docs1[i]), 'post_2_id': get_post_id(docs2[j]),
            'author_1': str(author1), 'author_2': str(author2),
            'topic_1': topics1[i], 'topic_2': topics2[j], 'same': False
        })

    df = pd.DataFrame(same_pairs + diff_pairs)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    data_path, out_path, num_pairs = sys.argv[1:]
    generate_pairs(data_path, out_path, int(num_pairs))
