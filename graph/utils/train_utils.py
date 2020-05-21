import time

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import TensorDataset
from tqdm import tqdm


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, sp_label, sent_start, sent_end, wd_edges, graph_mask,
                 sent_num, ques_edges=None, ad_edges=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sp_label = sp_label
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.wd_edges = wd_edges
        self.graph_mask = graph_mask
        self.sent_num = sent_num
        self.ques_edges = ques_edges
        self.ad_edges = ad_edges


def convert_examples_to_features(examples, label_list, max_query_length, max_seq_length, tokenizer, is_train=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):

        graph_mask = []
        sent_start = []
        sent_end = []
        sp_label = None

        if is_train:
            sp_label = []

        query_tokens = tokenizer.tokenize(example.questino)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        sent_tokens = []


        cur_pos = 0
        for si, sent in enumerate(example.answer):
            sent_start.append(cur_pos)
            cur_tokens = tokenizer.tokenize(sent)
            sent_tokens.extend(cur_tokens)
            sent_end.append(cur_pos + len(cur_tokens))
            cur_pos += len(cur_tokens)
            graph_mask.append(1)
            if is_train:
                sp_label.append(int(example.label[si]))

        # print(sent_tokens)
        # print(sent_start)
        # print(sent_end)
        # print(graph_mask)
        # print(sp_label)

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        cls_index = 0
        # print(len(tokens))
        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        # print(len(tokens))
        # SEP token
        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)
        # print(len(tokens))
        for token in sent_tokens:
            tokens.append(token)
            segment_ids.append(1)
        # print(len(tokens))
        # SEP token
        tokens.append(tokenizer.sep_token)
        segment_ids.append(1)
        # print(len(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]

        for si in range(len(sent_start)):
            sent_start[si] += len(query_tokens) + 2
            sent_end[si] += len(query_tokens) + 2

        sent_start = [i for i in sent_start if i < max_seq_length]
        sent_end = sent_end[:len(sent_start)]

        # print(sent_start)
        # print(sent_end)
        #
        # print(len(input_ids))
        # print(max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(sent_start) == len(sent_end)

        sent_num = len(sent_start)

        def wd_edge_list(data):
            """with-document edge list

            Arguments:
                data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            wd_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1i != s2i:
                        # if abs(s1i - s2i) <= 3:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        wd_edges = wd_edge_list(sent_start)

        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s" % " ".join([str(x) for x in sp_label]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sp_label=sp_label,
                          sent_start=sent_start,
                          sent_end=sent_end,
                          wd_edges=wd_edges,
                          graph_mask=graph_mask,
                          sent_num=sent_num
                          ))
    return features


def convert_examples_to_features_sent(examples, label_list, max_query_length, max_seq_length, tokenizer, is_train=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):

        graph_mask = []
        sent_start = []
        sent_end = []
        sp_label = None
        if is_train:
            sp_label = []

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
            if is_train:
                sp_label.append(int(example.label[si]))
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            segment_ids.append(sent_segment_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        sent_num = len(input_ids)

        # print(sent_tokens)
        # print(sent_start)
        # print(sent_end)
        # print(graph_mask)
        # print(sp_label)

        def wd_edge_list(data):
            """with-document edge list

            Arguments:
                data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            wd_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1i != s2i:
                        # if abs(s1i - s2i) <= 3:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        wd_edges = wd_edge_list(input_ids)
        # print(wd_edges)

        # content = tokenizer.tokenize(sent_text)
        # _truncate_seq_pair(query_tokens, content, max_seq_length - 3)
        #
        # # Feature ids
        # tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + content + ["[SEP]"]
        # segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(content) + 1)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #
        # # Mask and Paddings
        # input_mask = [1] * len(input_ids)
        # padding = [0] * (max_seq_length - len(input_ids))
        #
        # input_ids += padding
        # input_mask += padding
        # segment_ids += padding
        #
        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        #
        # if is_train:
        #     label_id = label_map[example.label[0]]
        # else:
        #     label_id = None

        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     # logger.info("tokens: %s" % " ".join(
        #     #             #     [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s" % " ".join([str(x) for x in sp_label]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sp_label=sp_label,
                          sent_start=sent_start,
                          sent_end=sent_end,
                          wd_edges=wd_edges,
                          graph_mask=graph_mask,
                          sent_num=sent_num
                          ))
    return features


def convert_examples_to_features_sent_ques(examples, label_list, max_query_length, max_seq_length, tokenizer,
                                           is_train=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):

        graph_mask = []
        sent_start = []
        sent_end = []
        sp_label = None
        if is_train:
            sp_label = []

        query_tokens = tokenizer.tokenize(example.question)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        ques_np_ety = example.ques_np_ety

        input_ids = []
        input_mask = []
        segment_ids = []
        cur_pos = 0
        for si, sent in enumerate(example.answer):
            sent_start.append(cur_pos)
            cur_tokens = tokenizer.tokenize(sent['text'])
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
            if is_train:
                sp_label.append(int(example.label[si]))
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            segment_ids.append(sent_segment_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        sent_num = len(input_ids)

        # print(sent_tokens)
        # print(sent_start)
        # print(sent_end)
        # print(graph_mask)
        # print(sp_label)

        def check_overlap(sent1, sent2):
            for ner1 in sent1:
                for ner2 in sent2:
                    if ner1 and ner2:
                        if ner1[0] == ner2[0]:
                            return True

            return False

        def check_ques(sent1, sent2, ques):
            for ner1 in sent1:
                for ner2 in sent2:
                    # print(ner1[0], ner2[0], ques)
                    # if ner1[0] != ner2[0] and ner1[0] in ques and ner2[0] in ques:
                    if ner1 and ner2 and ques:
                        # try:
                        if ner1[0] in ques and ner2[0] in ques:  # no matter same or not
                            return True
                        # except:
                        #     print(ques)
                        #     print(ner1)
                        #     print(ner2)
            return False

        def wd_edge_list(data):
            """with-document edge list

            Arguments:
                data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            wd_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1i != s2i and s1['doc_id'] == s2['doc_id']:
                        # if abs(s1i - s2i) <= 3:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        def ad_edge_list(data):
            """with-document edge list

            Arguments:
                data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """

            ad_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1['doc_id'] != s2['doc_id'] and check_overlap(s1['ans_np_ety'], s2['ans_np_ety']):
                        ad_edge_list.append([s1i, s2i])
            return ad_edge_list

        def ques_edge_list(data, ques):
            """connections with question NER and NP as bridge

            Arguments:
                data {[type]} -- [description]
            """
            ques_edge_list = []
            for di_idx, di in enumerate(data):
                for dj_idx, dj in enumerate(data):
                    if check_ques(di['ans_np_ety'], dj['ans_np_ety'], ques) and (di['doc_id'] != dj['doc_id']):
                        ques_edge_list.append([di_idx, dj_idx])
            return ques_edge_list

        wd_edges = wd_edge_list(example.answer)
        ques_edges = ques_edge_list(example.answer, ques_np_ety)
        ad_edges = ad_edge_list(example.answer)
        # print('*'*30)
        # print(wd_edges)
        # print(ques_edges)
        # print(ad_edges)

        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     # logger.info("tokens: %s" % " ".join(
        #     #             #     [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s" % " ".join([str(x) for x in sp_label]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sp_label=sp_label,
                          sent_start=sent_start,
                          sent_end=sent_end,
                          wd_edges=wd_edges,
                          graph_mask=graph_mask,
                          sent_num=sent_num,
                          ques_edges=ques_edges,
                          ad_edges=ad_edges
                          ))
    return features


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


def get_batches(features, is_train=False):
    start = time.time()

    def get_batch_stat(features):
        max_sent_num = 0
        for d in features:
            if len(d.sent_start) > max_sent_num:
                max_sent_num = len(d.sent_start)

        return max_sent_num

    max_nodes = get_batch_stat(features)
    # print(max_nodes)
    # batching
    minibatch_size = len(features)
    input_len = len(features[0].input_ids)
    input_ids = torch.zeros(minibatch_size, input_len, dtype=torch.long)
    input_mask = torch.zeros(minibatch_size, input_len, dtype=torch.long)
    segment_ids = torch.zeros(minibatch_size, input_len, dtype=torch.long)
    for fi, f in enumerate(features):
        input_ids[fi, :] = torch.tensor(f.input_ids, dtype=torch.long)
        input_mask[fi, :] = torch.tensor(f.input_mask, dtype=torch.long)
        segment_ids[fi, :] = torch.tensor(f.segment_ids, dtype=torch.long)
    input_graph_mask = torch.zeros(minibatch_size, max_nodes)
    if is_train:
        input_sp_label = torch.zeros(minibatch_size, max_nodes)
    sent_start = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)
    sent_end = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)

    adj_matrix = gen_adj_matrix(features, max_nodes,
                                wdedge=True, quesedge=False, adedge=False)

    # nodes configuration: [cands, docs, mentions, subs]
    for di, d in enumerate(features):
        for si in range(len(d.sent_start)):
            if d.sent_start[si] < input_len:
                input_graph_mask[di, si] = 1  # sent_mask
                if is_train:
                    input_sp_label[di, si] = d.sp_label[si]
            else:
                print("Some sentences in sample {} were cut!".format(d.example_index))
        sent_start[di, :len(d.sent_start)] = torch.tensor(d.sent_start)
        sent_end[di, :len(d.sent_start)] = torch.tensor(d.sent_end)

    if is_train:
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end,
                             input_sp_label)
    else:
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end)

    # all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    # if is_train:
    #     all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # else:
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    return data


def get_batches_sent(args, features, is_train=False, full_pas=False):
    start = time.time()

    def get_batch_stat(features):
        max_sent_num = 0
        for d in features:
            if d.sent_num > max_sent_num:
                max_sent_num = d.sent_num

        return max_sent_num

    max_nodes = get_batch_stat(features)
    print(max_nodes)
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
        # print(f.sent_num)
        # print(len(f.input_ids))
        input_ids[fi, :f.sent_num] = torch.tensor(f.input_ids, dtype=torch.long)
        input_mask[fi, :f.sent_num] = torch.tensor(f.input_mask, dtype=torch.long)
        segment_ids[fi, :f.sent_num] = torch.tensor(f.segment_ids, dtype=torch.long)
        # input_graph_mask[fi] = torch.tensor(f.graph_mask, dtype=torch.long)
        sent_num[fi] = torch.tensor(f.sent_num, dtype=torch.long)
    if is_train:
        input_sp_label = torch.zeros(minibatch_size, max_nodes)
    for di, d in enumerate(features):
        for si in range(d.sent_num):
            input_graph_mask[di, si] = 1  # sent_mask
            if is_train:
                input_sp_label[di, si] = d.sp_label[si]
    sent_start = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)
    sent_end = -1 * torch.ones(minibatch_size, max_nodes, dtype=torch.long)

    adj_matrix = gen_adj_matrix(features, max_nodes,
                                wdedge=args.wdedge, quesedge=args.quesedge, adedge=args.adedge)

    # nodes configuration: [cands, docs, mentions, subs]
    # for di, d in enumerate(features):
    #     for si in range(len(d.sent_start)):
    #         if d.sent_start[si] < input_len:
    #             input_graph_mask[di, si] = 1  # sent_mask
    #             if is_train:
    #                 input_sp_label[di, si] = d.sp_label[si]
    #         else:
    #             print("Some sentences in sample {} were cut!".format(d.example_index))
    #     sent_start[di, :len(d.sent_start)] = torch.tensor(d.sent_start)
    #     sent_end[di, :len(d.sent_start)] = torch.tensor(d.sent_end)

    if is_train:
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end,
                             input_sp_label, sent_num)
    else:
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end,
                             sent_num)

    # all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    # if is_train:
    #     all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # else:
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

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


class InputExample(object):

    def __init__(self, guid, question, answer=None, label=None, ques_np_ety=None):
        self.guid = guid
        self.question = question
        self.answer = answer
        self.label = label
        self.ques_np_ety = ques_np_ety
