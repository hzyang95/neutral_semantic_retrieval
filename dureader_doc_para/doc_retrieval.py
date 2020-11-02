import json
import os
import sys
from multiprocessing.pool import ThreadPool

import jieba
import jieba.analyse
import requests
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from dureader_doc_para.utils import parse_config, mkdata, subDataset, getData, getTestData, mktestdata

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
test_data = getTestData(conf_file)
# print(len(test_data))

eee = False
saved =[]

if os.path.exists('lda.json'):
    eee = True
    with open('lda.json', 'r', encoding='utf-8') as f:
        for item in f.readlines():
            _item = json.loads(item)
            saved.append(_item)
else:
    json_file = open('lda.json', 'w', encoding='utf-8')
ref = []
res = []

task = [False, False, False]

task[0] = True

kws = task[0]
lda = task[1]
full = task[2]
for i, sample in enumerate(tqdm(test_data[:])):
    tokens = sample[0]
    label = [item[2] for item in tokens]
    ress = []
    top = sample[1]
    if kws:
        for item in tokens:
            keywords = jieba.analyse.extract_tags(item[0], topK=5, withWeight=True, allowPOS=())
            # keywords = jieba.analyse.textrank(item[0], topK=5, withWeight=True, allowPOS=())

            # 访问提取结果
            keywords = {_item[0]: _item[1] for _item in keywords}
            # print(keywords)
            # keywords = list(jieba.cut(item[0]))
            query_k = jieba.analyse.extract_tags(item[1], topK=5, withWeight=True, allowPOS=())
            # print(item[1])
            # query_k = jieba.analyse.textrank(item[1], topK=5, withWeight=True)

            # print(query_k)

            query_k = {_item[0]: _item[1] for _item in query_k}

            num = 0
            for i in query_k:
                # if i in item[3]:
                if i in keywords:
                    num += query_k[i]*keywords[i]
                    # num += 1
            ress.append(num)
    elif full:
        for item in tokens:
            query = item[1]
            cont = item[0]
            query_words = list(jieba.cut(query))
            cont_words = list(jieba.cut(cont))
            fll = 0
            for word in query_words:
                if word in cont_words:
                    fll += 1
            ress.append(fll)

    elif lda:

        def proc(js):
            r1 = requests.post(url='http://39.98.138.178:44444/similarity/query',
                               params=js)
            try:
                cont = json.loads(r1.content)
                # num = cont['data'][0] + cont['data'][1]
                num = cont['data']
            except:
                num = [0, 0]
            return num

        cra = []
        for item in tokens:
            query = item[1]
            cont = item[0]
            # title = item[3]
            cra.append({'query': query, 'text': cont})
            # r1 = requests.post(url='http://39.98.138.178:44444/similarity/query',
            #                    params={'query': query,
            #                            'text': title+cont})
            # cont = json.loads(r1.content)
            # num = cont['data'][0]+cont['data'][1]
            # ress.append(num)

        if eee is False:
            with ThreadPool(len(cra)) as threads:
                crawl_results = threads.map(proc, cra)
                json_file.write(json.dumps(crawl_results) + '\n')
        else:
            crawl_results = saved[i]

        crawl_results = [i[0]+i[1] for i in crawl_results]

        print(crawl_results)
        ress = crawl_results
    else:
        text = [' '.join(jieba.cut(item[0])) for item in tokens]

        ques = [' '.join(jieba.cut(tokens[0][1]))]

        tokenized_tokens = text + ques
        tfidf = TfidfVectorizer()
        tfidf.fit(tokenized_tokens)
        x = tfidf.transform(text).toarray()
        y = tfidf.transform(ques).toarray()
        ress = np.matmul(x, y.transpose())
    # if not full:
    tops = torch.topk(torch.tensor(ress), min(3, len(tokens)), dim=0)
    ind = [0] * len(tokens)
    # print(tops)
    for ii in tops[1]:
        if kws or lda or full:
            ind[ii] = 1
        else:
            ind[ii[0]] = 1
    #     # print(ind)
    # else:
    #     ind = ress
    ref += label
    res += ind

assert len(ref) == len(res)

length = len(ref)
tp = 0
fp = 0
tn = 0
fn = 0

for i in range(length):
    a = ref[i]
    b = res[i]
    if a == 1:
        if b == 1:
            tp += 1
        else:
            fn += 1
    else:
        if b == 1:
            fp += 1
        else:
            tn += 1
# [28349, 17462]
#  18393, 9603]
# [26252, 19559]
# tp = 9603
# fp = 7859
# tn = 18393
# fn = 9956
print("tp: " + str(tp))
print("fp: " + str(fp))
print("tn: " + str(tn))
print("fn: " + str(fn))
acc = (tp + tn) / (tp + fp + tn + fn)
acc = round(acc, 4)
print("acc: " + str(acc))
pr = tp / (tp + fp)
pr = round(pr, 4)
print("pre: " + str(pr))
rec = tp / (tp + fn)
rec = round(rec, 4)
print("rec: " + str(rec))
f1 = 2 * pr * rec / (pr + rec)
f1 = round(f1, 4)
print("f1: " + str(f1))
