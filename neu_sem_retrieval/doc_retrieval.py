import os
import sys
import jieba
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from neu_sem_retrieval.utils import parse_config, mkdata, subDataset, getData,getTestData,mktestdata

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
test_data = getTestData(conf_file)
# print(len(test_data))
ref = []
res = []
for i, sample in enumerate(test_data[:200]):
    tokens = sample[0]
    text = [' '.join(jieba.cut(item[0])) for item in tokens]
    label = [item[2] for item in tokens]
    ques = [' '.join(jieba.cut(tokens[0][1]))]
    top = sample[1]

    # print(ques)
    # print(label)
    # print(text)
    tokenized_tokens = text + ques

    tfidf = TfidfVectorizer()
    tfidf.fit(tokenized_tokens)
    x = tfidf.transform(text).toarray()
    y = tfidf.transform(ques).toarray()
    # print(x.shape)
    # print('='*20)
    # print(y.shape)
    output = np.matmul(x, y.transpose())
    # print(res)
    # print(np.argmax(res))
    tops = torch.topk(torch.tensor(output), min(4, len(text)), dim=0)
    ind = [0] * len(text)

    for ii in tops[1]:
        ind[ii[0]] = 1
    # print(ind)
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
    if a==1:
        if b==1:
            tp += 1
        else:
            fn += 1
    else:
        if b==1:
            fp += 1
        else:
            tn += 1

print("tp: "+str(tp))
print("fp: " + str(fp))
print("tn: " + str(tn))
print("fn: " + str(fn))
print("acc: " + str((tp+tn) / (tp + fp + tn + fn)))
pr = tp/(tp+fp)
print("pre: "+ str(pr))
rec = tp / (tp + fn)
print("rec: " + str(rec))
f1 = 2*pr*rec/(pr+rec)
print("f1:"+ str(f1))