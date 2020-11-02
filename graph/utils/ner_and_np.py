import json
import logging
import os
import sys
import time

import requests
from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP

# nlp = StanfordCoreNLP(r'/users8/hzyang/stanford-corenlp-full-2020-04-20/', memory='8g', port=56789, lang='zh')
# , quiet=False, logging_level=logging.DEBUG

# sentence = ['哈尔滨工业大学位于哈尔滨，工业很强。', '这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。']

from pyltp import NamedEntityRecognizer, Postagger, Segmentor

MODELDIR = '/users8/hzyang/proj/neutral_semantic_retrieval/ltp_data_v3.4.0'
# MODELDIR = 'ltp_data_v3.4.0'

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))
recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))
postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))


# print(nlp.word_tokenize(sentence))
# print(nlp.pos_tag(sentence))

def _request(annotators=None, data=None):
    if sys.version_info.major >= 3:
        data = data.encode('utf-8')

    properties = {'annotators': annotators, 'outputFormat': 'json'}
    params = {'properties': str(properties), 'pipelineLanguage': 'zh'}

    logging.info(params)
    # http://39.98.138.178:44444
    # http://127.0.0.1:56789
    r = requests.post('http://127.0.0.1:56789', params=params, data=data, headers={'Connection': 'close'})
    try:
        r_dict = json.loads(r.text)
    except:
        r_dict = {}
    return r_dict


def ner(sentence):
    r_dict = _request('ner', sentence)
    words = []
    ner_tags = []
    try:
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['originalText'])
                ner_tags.append(token['ner'])
    except:
        print('ner')
        print(sentence)
    return list(zip(words, ner_tags))


def parse(sentence):
    r_dict = _request('pos,parse', sentence)
    # print(r_dict)
    res = []
    try:
        res = [s['parse'] for s in r_dict['sentences']][0]
    except:
        print('parse')
        print(sentence)
    return res


# nlp = StanfordCoreNLP('http://localhost', port=56789)


def get_NP(sent):
    pa = parse(sent)
    nps = []
    try:
        tree = Tree.fromstring(pa)
        for i in tree.subtrees():
            if i.label() == 'NP':
                nps.append(''.join(i.leaves()))
    except:
        print('get np')
        print(pa)
    return list(set(nps))


def get_NER(sent):
    ne = ner(sent)
    # print(ne)
    re = []
    try:
        for i in ne:
            if i[1] in ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'TIME']:
                re.append(i[0])
    except:
        print('get ner')
        print(ne)
    return list(set(re))


def get_NER_LTP(sent):
    res = []
    sent = segmentor.segment(sent)
    postags = postagger.postag(sent)
    # print("\t".join(postags))
    netags = recognizer.recognize(sent, postags)
    temp = ''
    for ind, i in enumerate(netags):
        i = i[0]
        if i == 'O':
            continue
        elif i == 'S':
            res.append(sent[ind])
        elif i == 'B':
            temp = sent[ind]
        elif i == 'I':
            temp += sent[ind]
        elif i == 'E':
            temp += sent[ind]
            res.append(temp)
    # print("\t".join(netags))
    return list(set(res))
