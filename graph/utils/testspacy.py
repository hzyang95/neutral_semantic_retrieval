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

sentence = ['哈尔滨工业大学位于哈尔滨，工业很强。',
            '这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。这两种方法都需要提前下载CoreNLP最新的压缩包，再下载对应的语言jar包。']

from pyltp import NamedEntityRecognizer, Postagger, Segmentor

MODELDIR = '../../ltp_data_v3.4.0'

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
    r = requests.post('http://127.0.0.1:56789', params=params, data=data, headers={'Connection': 'close'})
    r_dict = json.loads(r.text)

    return r_dict


def ner(sentence):
    r_dict = _request('ner', sentence)
    words = []
    ner_tags = []
    for s in r_dict['sentences']:
        for token in s['tokens']:
            words.append(token['originalText'])
            ner_tags.append(token['ner'])
    return list(zip(words, ner_tags))


def parse(sentence):
    r_dict = _request('pos,parse', sentence)
    # print(r_dict)
    return [s['parse'] for s in r_dict['sentences']][0]


def get_NP(sent):
    pa = parse(sent)
    tree = Tree.fromstring(pa)
    nps = []
    for i in tree.subtrees():
        if i.label() == 'NP':
            nps.append(''.join(i.leaves()))
    return list(set(nps))


def get_NER(sent):
    ne = ner(sent)
    # print(ne)
    re = []
    for i in ne:
        if i[1] in ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'TIME']:
            re.append(i[0])
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

t1=0
t2=0
t3=0
for i in range(100):
    for sent in sentence:
        time1 = time.time()
        print(get_NP(sent))
        time2 = time.time()
        print(get_NER(sent))
        time3 = time.time()
        print(get_NER_LTP(sent))
        time4 = time.time()
        t1 +=time2 - time1
        t2 +=time3 - time2
        t3 += time4 - time3
        print(t1)
        print(t2)
        print(t3)

# print(nlp.dependency_parse(sentence))


# for sent in sentence:
#
#     sent = segmentor.segment(sent)
#     postags = postagger.postag(sent)
#     print("\t".join(postags))
#     netags = recognizer.recognize(sent, postags)
#     print("\t".join(netags))


'''
{'sentences': 
    [
        {
            'index': 0, 
            'parse': '(ROOT\n  (IP\n    (NP (NR 清华) (NN 大学))\n    (VP (VV 位于)\n      (NP (NR 北京)))\n    (PU 。)))', 
            'binaryParse': '(ROOT\n  (IP\n    (NP (NR 清华) (NN 大学))\n    (@IP\n      (VP (VV 位于)\n        (NP (NR 北京)))\n      (PU 。))))', 
            'basicDependencies': [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 3, 'dependentGloss': '位于'}, {'dep': 'compound:nn', 'governor': 2, 'governorGloss': '大学', 'dependent': 1, 'dependentGloss': '清华'}, {'dep': 'nsubj', 'governor': 3, 'governorGloss': '位于', 'dependent': 2, 'dependentGloss': '大学'}, {'dep': 'dobj', 'governor': 3, 'governorGloss': '位于', 'dependent': 4, 'dependentGloss': '北京'}, {'dep': 'punct', 'governor': 3, 'governorGloss': '位于', 'dependent': 5, 'dependentGloss': '。'}], 
            'enhancedDependencies': [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 3, 'dependentGloss': '位于'}, {'dep': 'compound:nn', 'governor': 2, 'governorGloss': '大学', 'dependent': 1, 'dependentGloss': '清华'}, {'dep': 'nsubj', 'governor': 3, 'governorGloss': '位于', 'dependent': 2, 'dependentGloss': '大学'}, {'dep': 'dobj', 'governor': 3, 'governorGloss': '位于', 'dependent': 4, 'dependentGloss': '北京'}, {'dep': 'punct', 'governor': 3, 'governorGloss': '位于', 'dependent': 5, 'dependentGloss': '。'}], 
            'enhancedPlusPlusDependencies': [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 3, 'dependentGloss': '位于'}, {'dep': 'compound:nn', 'governor': 2, 'governorGloss': '大学', 'dependent': 1, 'dependentGloss': '清华'}, {'dep': 'nsubj', 'governor': 3, 'governorGloss': '位于', 'dependent': 2, 'dependentGloss': '大学'}, {'dep': 'dobj', 'governor': 3, 'governorGloss': '位于', 'dependent': 4, 'dependentGloss': '北京'}, {'dep': 'punct', 'governor': 3, 'governorGloss': '位于', 'dependent': 5, 'dependentGloss': '。'}], 
            'tokens': [
                {'index': 1, 'word': '清华', 'originalText': '清华', 'characterOffsetBegin': 0, 'characterOffsetEnd': 2, 'pos': 'NR'}, 
                {'index': 2, 'word': '大学', 'originalText': '大学', 'characterOffsetBegin': 2, 'characterOffsetEnd': 4, 'pos': 'NN'}, 
                {'index': 3, 'word': '位于', 'originalText': '位于', 'characterOffsetBegin': 4, 'characterOffsetEnd': 6, 'pos': 'VV'}, 
                {'index': 4, 'word': '北京', 'originalText': '北京', 'characterOffsetBegin': 6, 'characterOffsetEnd': 8, 'pos': 'NR'}, 
                {'index': 5, 'word': '。', 'originalText': '。', 'characterOffsetBegin': 8, 'characterOffsetEnd': 9, 'pos': 'PU'}
            ]
        }
    ]
}




'''
