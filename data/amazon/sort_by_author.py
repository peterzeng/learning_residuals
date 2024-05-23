import json
import random


def read_data(author_to_docs, data_path, topic):
    min_words = 20

    with open(data_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            author_id = json_data['reviewerID']

            if 'reviewText' in json_data:
                text = json_data['reviewText']
                if len(text.split()) < min_words:
                    continue

                if author_id not in author_to_docs:
                    author_to_docs[author_id] = {'syms': [], 'topic': [], 'author_id': author_id}

                author_to_docs[author_id]['syms'].append(json_data['reviewText'])
                author_to_docs[author_id]['topic'].append(topic)


def save_to_json(data_list, out_path):
    with open(out_path, 'w') as outfile:
        for entry in data_list:
            json.dump(entry, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    author_to_docs = {}
    read_data(author_to_docs, 'Office_Products_5.json', 'Office Products')
    read_data(author_to_docs, 'Patio_Lawn_and_Garden_5.json', 'Patio Lawn and Garden')
    read_data(author_to_docs, 'Video_Games_5.json', 'Video Games')

    data_list = []
    for author_id, author_dict in author_to_docs.items():
        if len(author_dict['syms']) >= 2:
            data_list.append({
                'syms': author_dict['syms'],
                'topic': author_dict['topic'],
                'author_id': author_id
            })

    random.shuffle(data_list)

    train_split = len(data_list) // 7 * 5
    dev_split = len(data_list) // 7 * 6

    save_to_json(data_list[:train_split], 'train_processed.jsonl')
    save_to_json(data_list[train_split:dev_split], 'dev_processed.jsonl')
    save_to_json(data_list[dev_split:], 'test_processed.jsonl')
