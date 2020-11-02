import time
import datetime as dt

import tornado.gen
import tornado.httpserver
import tornado.ioloop
import tornado.web
import json
import json
import numpy as np
from data_processor import DataProcessor
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import Counter
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def get_test_data(example):
    row = example['sample']

    samples = []
    for i,item in enumerate(row):
        guid = "%s-%s" % ('test', i)
        text_a = item['question']
        # answer = '{} {}'.format(row['context'], row['title'])
        text_b = '{}'.format(item['text'])
        label = item['label']
        samples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return samples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features_test(example, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""


    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(example):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # Feature ids
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Mask and Paddings
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5 and verbose:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def evaluate(model, feature, top):
    input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
    # data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    # Run prediction for full data
    # sampler = SequentialSampler(data)
    b_l = input_ids.size(0)
    # dataloader = DataLoader(data, sampler=sampler, batch_size=b_l)
    model.eval()
    # for input_ids, input_mask, segment_ids, label_ids in dataloader: #, desc="Evaluation")
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=input_mask)[0]
    sigmoid = torch.nn.Sigmoid()
    logits = sigmoid(logits)
    tops = torch.topk(logits, min(top, b_l), dim=0)
    ind = [0] * b_l
    for ii in tops[1]:
        ind[ii[0]] = 1
    return ind


class RetriHandler(tornado.web.RequestHandler):

    model = None
    tokenizer = None
    def initialize(self):
        path = '1_para_200000_1000_Evaluation_epoch2_step6000_f10.6517_loss0.1892.bin'
        model_state_dict = torch.load('checkpoints/' + path)  # args.ckpt_
        if not RetriHandler.tokenizer:
            RetriHandler.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased', do_lower_case=True)
        if not RetriHandler.model:
            RetriHandler.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', state_dict=model_state_dict,
                                                                    num_labels=1)
            RetriHandler.model.to('cpu')

    @tornado.gen.coroutine
    def post(self):
        received = json.loads(self.request.body)

        gra = received['gra']
        if gra == 'para':
            max_seq_length = 200
        else:
            max_seq_length = 50
        # time.sleep(5)
        # yield tornado.gen.sleep(3)
        data_test = get_test_data(received)
        label_list = [False, True]
        features = convert_examples_to_features_test(
            data_test, label_list, max_seq_length, self.tokenizer, verbose=False)
        ref = evaluate(self.model, features, 3)
        self.write({'res': ref})


if __name__ == "__main__":
    app = tornado.web.Application(
        [
            (r"/", RetriHandler)
        ],
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(56789)
    print('server start')
    tornado.ioloop.IOLoop.instance().start()
