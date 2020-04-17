import json
import os
import random
import re
import sys
import time

import jieba
from tqdm import tqdm

from os.path import dirname, abspath

path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)

random.seed(123456)

# f = 'model.pt'
# st = StanceDetector(f, True)


def save_file(dateset, task, _dict):
    dir_path = "processed/" + task + '_' + str(len(dateset)) + '_' + str(_dict[0]) + '_' + str(
        _dict[1])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    random.shuffle(dateset)
    questions = []
    answers = []
    labels = []
    time1 = time.time()
    for i in tqdm(range(len(dateset))):
        item = dateset[i]
        label = str(item['label'])
        question = item['question'].replace('\n', ' ')
        answer = item['text'].replace('\n', ' ')
        labels.append(label)
        questions.append(question)
        answers.append(answer[:520])
    time2 = time.time()
    print(time2 - time1)
    with open(dir_path + '/labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
    with open(dir_path + '/answers.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(answers))
    with open(dir_path + '/questions.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(questions))


if __name__ == "__main__":
    task = "docretrifull"
    # task = "newsimmixaver"
    aug_data_file = '../../preprocessed/version3_doc_full_train_1285618.json'
    test_data_file = '../../preprocessed/version3_doc_full_dev_45811.json'
    otf = 1


    # train_set = []
    # ll = 0
    # _dict = {1: 0, 0: 0}
    # with open(aug_data_file, "r", encoding='utf-8') as f:
    #     for item in f.readlines():
    #         _item = json.loads(item)
    #         # if len(_item['answer'])>100:
    #         #     continue
    #         ll += len(_item['text'][:520])
    #         _dict[_item['label']] += 1
    #         train_set.append(_item)
    # print(len(train_set))
    # print(_dict)
    # print(ll/len(train_set))
    # if otf:
    #     save_file(train_set, task + '_train', _dict)

    test_set = []
    ll = 0
    _dict = {1: 0, 0: 0}
    with open(test_data_file, "r", encoding='utf-8') as f:
        for item in f.readlines():
            _item = json.loads(item)
            # if len(_item['answer'])>100:
            #     continue
            ll += len(_item['text'][:520])
            _dict[_item['label']] += 1
            test_set.append(_item)
    print(len(test_set))
    print(_dict)
    print(ll / len(test_set))
    if otf:
        save_file(test_set, task + '_test', _dict)
