import time

import torch
from scipy.sparse import coo_matrix
from torch.utils.data import TensorDataset
import numpy as np

def load_torch_model(model_path, use_cuda=True):
    with open(model_path + "/model.pt", "rb") as f:
        if use_cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
            model.cpu()
        return model


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
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label)
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

def get_batches_sent(features, is_train=False):
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
                                wdedge=True, quesedge=False, adedge=False)

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
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label,sent_num)
    else:
        data = TensorDataset(input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end,sent_num)

    # all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    # if is_train:
    #     all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # else:
    #     data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    return data
