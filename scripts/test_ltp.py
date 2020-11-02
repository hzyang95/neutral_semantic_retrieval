#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

# ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
# sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path
#
# # Set your own model path
# MODELDIR=os.path.join(ROOTDIR, "./ltp_data_v3.4.0")
#
# from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
#
# # paragraph = '中国进出口银行与中国银行加强合作。中国进出口银行与中国银行加强合作！'
# paragraph = '双黄连能治疗新冠肺炎吗？'
#
# sentence = SentenceSplitter.split(paragraph)[0]
#
# segmentor = Segmentor()
# segmentor.load(os.path.join(MODELDIR, "cws.model"))
# words = segmentor.segment(sentence)
#
# import jieba
#
# words=list(jieba.cut(sentence))
# print("\t".join(words))
#
# postagger = Postagger()
# postagger.load(os.path.join(MODELDIR, "pos.model"))
# postags = postagger.postag(words)
# # list-of-string parameter is support in 0.1.5
# # postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
# print("\t".join(postags))
#
# parser = Parser()
# parser.load(os.path.join(MODELDIR, "parser.model"))
# arcs = parser.parse(words, postags)
#
# print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
#
# recognizer = NamedEntityRecognizer()
# recognizer.load(os.path.join(MODELDIR, "ner.model"))
# netags = recognizer.recognize(words, postags)
# print("\t".join(netags))
#
# labeller = SementicRoleLabeller()
# labeller.load(os.path.join(MODELDIR, "pisrl.model"))
# roles = labeller.label(words, postags, arcs)
#
# for role in roles:
#     print(role.index, "".join(
#             ["%s:(%d,%d) " % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
#
# segmentor.release()
# postagger.release()
# parser.release()
# recognizer.release()
# labeller.release()

# from pyhanlp import *
# # paragraph = '双黄连能治疗新冠肺炎吗？'
# document = "双黄连能治疗新冠肺炎吗？"
# print(HanLP.extractKeyword(document, 3))
#
# import time
#
# import requests
# import json
#
# text = [
#     '在确认无人感染后，球队将按照海南省疫情防控工作指挥部关于中国国家男子足球队封闭隔离转训工作安排，并报三亚市疫情防控工作指挥部获批，球队于26'
#     '日开始封闭性训练。在集中医学观察期结束前，全队还将再次进行核酸检测，以确保全体人员健康安全。',
#     '在外交部出台了暂停外国人持目前有效签证入境的通知后，很多中超、中甲俱乐部都已经通知了还在国外的队中外援，先暂时留在各自家中，做好个人和全家的防护工作，等待俱乐部的通知，如果可以入境的时候再返回中国。'
# ]
# for t in text:
#     time1 = time.time()
#     r1 = requests.post(url='http://39.98.138.178:44444/similarity/query',
#                        params={'query': '国足报平安!昨日开始封闭性训练 此后还将接受检测',
#                                'text': t})
#     time2 = time.time()
#
#     print(time2 - time1)
#
#     print(r1)
#
#     cont = json.loads(r1.content)
#     print(cont)

print(zip)