import time
import datetime as dt

import tornado.gen
import tornado.httpserver
import tornado.ioloop
import tornado.web
import json
import json
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import Counter
from transformers import DistilBertTokenizer, BertTokenizer
from transformers import DistilBertForSequenceClassification


class InputExample(object):

    def __init__(self, guid, question, answer=None, ques_np_ety=None):
        self.guid = guid
        self.question = question
        self.answer = answer
        self.ques_np_ety = ques_np_ety


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, sent_start, sent_end, wd_edges, graph_mask,
                 sent_num, ques_edges=None, ad_edges=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.wd_edges = wd_edges
        self.graph_mask = graph_mask
        self.sent_num = sent_num
        self.ques_edges = ques_edges
        self.ad_edges = ad_edges


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


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def create_examples(example):
    examples = []
    row = example
    ll = 0 
    question = row['question']
    content = []
    for item in row['sample']:
        lll = len(item['text'])
        if check_contain_chinese(item['text']) is False:
            continue
        if lll < 5:
            continue
        if lll > 200:
            continue
        if len(content) + 1 >= 50:
            examples.append(InputExample(question=question, answer=content))
            content = []
            ll = 0
        ll += lll
        content.append(item['text'])
    examples.append(InputExample(question=question, answer=content))
    return examples


def convert_examples_to_features_sent(examples, max_query_length, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(examples):

        graph_mask = []
        sent_start = []
        sent_end = []

        query_tokens = tokenizer.tokenize(example.question)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        input_ids = []
        input_mask = []
        segment_ids = []
        cur_pos = 0
        for si, sent in enumerate(example.answer):
            sent_start.append(cur_pos)
            cur_tokens = tokenizer.tokenize(sent)
            _truncate_seq_pair(query_tokens, cur_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + cur_tokens + ["[SEP]"]
            sent_segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(cur_tokens) + 1)
            sent_input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Mask and Paddings
            sent_input_mask = [1] * len(sent_input_ids)
            sent_padding = [0] * (max_seq_length - len(sent_input_ids))

            sent_input_ids += sent_padding
            sent_input_mask += sent_padding
            sent_segment_ids += sent_padding

            assert len(sent_input_ids) == max_seq_length
            assert len(sent_input_mask) == max_seq_length
            assert len(sent_segment_ids) == max_seq_length

            graph_mask.append(1)

            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            segment_ids.append(sent_segment_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        sent_num = len(input_ids)

        def wd_edge_list(data):
            wd_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1i != s2i:
                        # if abs(s1i - s2i) <= 3:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        wd_edges = wd_edge_list(input_ids)
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sent_start=sent_start,
                          sent_end=sent_end,
                          wd_edges=wd_edges,
                          graph_mask=graph_mask,
                          sent_num=sent_num
                          ))
    return features


def get_batches_sent(features):
    def get_batch_stat(features):
        max_sent_num = 0
        for d in features:
            if d.sent_num > max_sent_num:
                max_sent_num = d.sent_num

        return max_sent_num

    max_nodes = get_batch_stat(features)
    # batching
    minibatch_size = len(features)
    input_len = len(features[0].input_ids)
    max_sent_len = len(features[0].input_ids[0])
    input_ids = torch.zeros(minibatch_size, max_nodes, max_sent_len, dtype=torch.long)
    input_mask = torch.zeros(minibatch_size, max_nodes, max_sent_len, dtype=torch.long)
    segment_ids = torch.zeros(minibatch_size, max_nodes, max_sent_len, dtype=torch.long)
    input_graph_mask = torch.zeros(minibatch_size, max_nodes)
    sent_num = torch.zeros(minibatch_size)
    for fi, f in enumerate(features):
        input_ids[fi, :f.sent_num] = torch.tensor(f.input_ids, dtype=torch.long)
        input_mask[fi, :f.sent_num] = torch.tensor(f.input_mask, dtype=torch.long)
        segment_ids[fi, :f.sent_num] = torch.tensor(f.segment_ids, dtype=torch.long)
        # input_graph_mask[fi] = torch.tensor(f.graph_mask, dtype=torch.long)
        sent_num[fi] = torch.tensor(f.sent_num, dtype=torch.long)
    for di, d in enumerate(features):
        for si in range(d.sent_num):
            input_graph_mask[di, si] = 1  # sent_mask
    sent_start = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)
    sent_end = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)

    adj_matrix = gen_adj_matrix(features, max_nodes,
                                wdedge=True, quesedge=False, adedge=False)

    data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end,
                         sent_num)

    return data


def gen_adj_matrix(train_features, max_sent_num, adj_norm=False, wdedge=True, adedge=False, quesedge=False):
    def adj_proc(adj):
        adj = adj.numpy()
        d_ = adj.sum(-1)
        d_[np.nonzero(d_)] **= -1
        return torch.tensor(adj * np.expand_dims(d_, -1))

    # edge list
    adj_matrix = []
    for f in train_features:
        adj_tmp = []
        # print(f.wd_edges)
        if wdedge:
            if len(f.wd_edges) == 0:
                wd_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                wd_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.wd_edges)), np.array(f.wd_edges).T),
                                                       shape=(max_sent_num, max_sent_num)).toarray(), dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(wd_edges_tmp))
            else:
                adj_tmp.append(wd_edges_tmp)
            # wd_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            # if adj_norm:
            #     adj_tmp.append(adj_proc(wd_edges_tmp))
            # else:
            #     adj_tmp.append(wd_edges_tmp)
        if adedge:
            if len(f.ad_edges) == 0:
                ad_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                ad_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.ad_edges)), np.array(f.ad_edges).T),
                                                       shape=(max_sent_num, max_sent_num)).toarray(), dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(ad_edges_tmp))
            else:
                adj_tmp.append(ad_edges_tmp)
        if quesedge:
            if len(f.ques_edges) == 0:
                ques_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                ques_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.ques_edges)), np.array(f.ques_edges).T),
                                                         shape=(max_sent_num, max_sent_num)).toarray(),
                                              dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(ques_edges_tmp))
            else:
                adj_tmp.append(ques_edges_tmp)

        adj_matrix.append(torch.stack(adj_tmp, dim=0))
    adj_matrix = torch.stack(adj_matrix)

    return adj_matrix


def process_logit(batch_index, batch_logits,_sent_len, threshold):
    """get predictions for each sample in the batch

    Arguments:
        batch_index {[type]} -- [description]
        batch_logits {[type]} -- 0: supporting facts logits, 1: answer span logits, 2: answer type logits 3: gold doc logits
        batch_size {[type]} -- [description]
        predict_file {[type]} -- [description]
    """

    sp_logits_np = torch.sigmoid(batch_logits).detach().cpu().numpy()

    batch_index = batch_index.numpy().tolist()

    sp_pred = []

    for idx, data in enumerate(batch_index):
        sp_pred += sp_logits_np[idx, :].tolist()

    return sp_pred


def test(model, tokenizer, test_data, threshold):
    # model = model.module
    _ri = []
    _pr = []
    # return
    all_step = 0
    test_data = DataLoader(test_data, batch_size=50)
    with torch.no_grad():
        for step, batch in enumerate(test_data):
            # if step < 5:
            #     print(len(batch))
            batch = tuple(t.cpu() for t in batch)
            input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, sent_num = batch

            inputs = {'input_ids': input_ids,
                      'input_mask': input_mask,
                      'segment_ids': segment_ids,  # XLM don't use segment_ids
                      'adj_matrix': adj_matrix,
                      'graph_mask': input_graph_mask,
                      'sent_start': sent_start,
                      'sent_end': sent_end,
                      'sent_sum_way': 'avg',
                      'sent_num': sent_num,
                      'gtem': 'sent',
                      }
            _sent_len = []
            example_indices = torch.arange(all_step, all_step + batch[0].size(0))
            all_step += batch[0].size(0)
            model.eval()
            outputs = model(**inputs)
            pred_batch = process_logit(example_indices, outputs, _sent_len, threshold)
            _pr.extend(pred_batch)
    _pr = list(_pr)
    res = sorted(enumerate(_pr), key=lambda _pr: _pr[1], reverse=True)
    return [i[0] for i in res]

def load_torch_model(model_path, use_cuda=False):
    with open(model_path + "/model.pt", "rb") as f:
        if use_cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
            model.cpu()
        return model


class RetriHandler(tornado.web.RequestHandler):
    model = None
    tokenizer = None

    def initialize(self):
        path = 'checkpoints/sent_single_cmrc_proceed_lr2e-5_final_48_hfl_rbt3_11_2544_0017_9106'
        if not RetriHandler.tokenizer:
            RetriHandler.tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')
        if not RetriHandler.model:
            RetriHandler.model = load_torch_model(path, use_cuda=False).module


    @tornado.gen.coroutine
    def post(self):
        received = json.loads(self.request.body)

        gra = received['gra']
        # time.sleep(5)
        # yield tornado.gen.sleep(3)
        dt = create_examples(received)
        features = convert_examples_to_features_sent(
            dt, 50, 150, self.tokenizer)
        data = get_batches_sent(features)
        ref = test(self.model, self.tokenizer, data, 0.4)

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
