from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import DistilBertPreTrainedModel, DistilBertModel

from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings


class gcnLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, gcn_num_rel=3, batch_norm=False, edp=0.0):
        super(gcnLayer, self).__init__()
        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.gcn_num_rel = gcn_num_rel
        self.dropout = dropout
        self.edge_dropout = nn.Dropout(edp)

        for i in range(gcn_num_rel):
            setattr(self, "fr{}".format(i + 1),
                    nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False)))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False))

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim))

        self.act = GeLU()

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop):
            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i + 1))(cur_input) for i in range(self.gcn_num_rel)],
                                    1) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x 2 x max_nodes x node_dim

            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(self.edge_dropout(adj.float()), nb_output), dim=1, keepdim=False) + \
                     self.fs(cur_input) * input_mask.unsqueeze(-1)  # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * self.act(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class DisGraphBasedModel(DistilBertPreTrainedModel):

    def __init__(self, config, num_answer_type=3, num_hop=3, num_rel=2, no_gnn=False, edp=0.0,
                 sent_with_cls=False):
        super(DisGraphBasedModel, self).__init__(config)
        self.bert = DistilBertModel(config)
        # self.model_freeze()

        self.dropout = 0.1
        self.num_rel = num_rel
        self.config = config
        self.sent_with_cls = sent_with_cls
        self.no_gnn = no_gnn

        # if not self.no_gnn:
        #     self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,edp=edp)

        self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,
                                 edp=edp)

        self.hidden_size = int(config.hidden_size / 2)

        self.ori_hidden = int(config.hidden_size)

        self.sent_selfatt = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(),
                                          nn.Dropout(self.dropout), nn.Linear(self.hidden_size, 1))

        self.sp_classifier = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(),
                                           nn.Dropout(self.dropout),
                                           nn.Linear(self.hidden_size, 1))  # input: graph embeddings

        self.num_answer_type = num_answer_type
        self.sfm = nn.Softmax(-1)
        # self.answer_type_classifier.half()

        self.init_weights()

    def attention(self, x, z):
        # x: batch_size X max_nodes X feat_dim
        # z: attention logits

        att = self.sfm(z).unsqueeze(-1)  # batch_size X max_nodes X 1

        output = torch.bmm(att.transpose(1, 2), x)

        return output

    def gen_mask(self, max_len, lengths, device):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num,
                                                                    -1) + 1  # +1 for masking out sequences with length 0
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask

    # self attentive pooling
    def do_selfatt(self, input, input_len, selfatt, span_logits=None):

        # input: max_len X batch_size X dim

        input_mask = self.gen_mask(input.size(0), input_len, input.device)

        att = selfatt.forward(input).squeeze(-1).transpose(0, 1)
        att = att.masked_fill(input_mask, -9e15)
        if span_logits is not None:
            att = att + span_logits
        att_sfm = self.sfm(att).unsqueeze(1)

        # print(att_sfm[56:63,:,:])
        # exit()

        output = torch.bmm(att_sfm, input.transpose(0, 1)).squeeze(1)  # batchsize x dim

        return output

    def forward(self, input_ids, input_mask, segment_ids, adj_matrix, graph_mask, sent_start,
                sent_end, sp_label=None, sent_sum_way='attn', ):

        """
        input_ids: bs X num_doc X num_sent X sent_len
        token_type_ids: same size as input_ids
        attention_mask: same size as input_ids
        input_adj_matrix: bs X 3 X max_nodes X max_nodes
        input_graph_mask: bs X max_nodes
        """

        # Roberta doesn't use token_type_ids cause there is no NSP task
        segment_ids = torch.zeros_like(segment_ids).to(input_ids.device)

        # reshaping
        bs, sent_len = input_ids.size()
        max_nodes = adj_matrix.size(-1)

        outputs = self.bert(input_ids, attention_mask=input_mask)

        sequence_output = outputs[0]
        cls_output = outputs[0][0, :, :].unsqueeze(0)

        feat_dim = cls_output.size(-1)

        # sentence extraction
        per_sent_len = sent_end - sent_start
        max_sent_len = torch.max(sent_end - sent_start)
        if self.sent_with_cls:
            per_sent_len += 1
            max_sent_len += 1
        # print("Maximum sent length is {}".format(max_sent_len))
        sent_output = torch.zeros(bs, max_nodes, max_sent_len, feat_dim).to(input_ids.device)
        for i in range(bs):
            for j in range(max_nodes):
                if sent_end[i, j] <= sent_len:
                    if sent_start[i, j] != -1 and sent_end[i, j] != -1:
                        if not self.sent_with_cls:
                            sent_output[i, j, :(sent_end[i, j] - sent_start[i, j]), :] = sequence_output[i,
                                                                                         sent_start[i, j]:sent_end[
                                                                                             i, j], :]
                        else:
                            sent_output[i, j, 1:(sent_end[i, j] - sent_start[i, j]) + 1, :] = sequence_output[i,
                                                                                              sent_start[i, j]:sent_end[
                                                                                                  i, j], :]
                            sent_output[i, j, 0, :] = cls_output[i]
                else:
                    if sent_start[i, j] < sent_len:
                        if not self.sent_with_cls:
                            sent_output[i, j, :(sent_len - sent_start[i, j]), :] = sequence_output[i,
                                                                                   sent_start[i, j]:sent_len, :]
                        else:
                            sent_output[i, j, 1:(sent_len - sent_start[i, j]) + 1, :] = sequence_output[i,
                                                                                        sent_start[i, j]:sent_len, :]
                            sent_output[i, j, 0, :] = cls_output[i]

        # sent summarization
        if sent_sum_way == 'avg':
            sent_sum_output = sent_output.mean(dim=2)
        elif sent_sum_way == 'attn':
            sent_sum_output = self.do_selfatt(
                sent_output.contiguous().view(bs * max_nodes, max_sent_len, self.config.hidden_size).transpose(0,
                                                                                                               1), \
                per_sent_len.view(bs * max_nodes), self.sent_selfatt).view(bs, max_nodes, -1)

        # graph reasoning
        if not self.no_gnn and self.num_rel > 0:
            gcn_output = self.sp_graph(sent_sum_output, graph_mask, adj_matrix)  # bs X max_nodes X feat_dim
        else:
            # gcn_output = self.sp_graph(sent_sum_output, graph_mask, torch.zeros(bs,1,max_nodes,max_nodes).to(input_ids.device))
            gcn_output = sent_sum_output

        # sp sent classification
        sp_logits = self.sp_classifier(gcn_output).view(bs, max_nodes)
        sp_logits = torch.where(graph_mask > 0, sp_logits, -9e15 * torch.ones_like(sp_logits).to(input_ids.device))

        # print(sp_logits)
        # select top 10 sentences with highest logits and then recalculate start and end logits

        # answer type logits

        if sp_label is not None:
            return self.loss_func(sp_logits, sp_label), sp_logits
        else:
            return sp_logits

    def loss_func(self, sp_logits, sp_label):

        bce_crit = torch.nn.BCELoss()
        ce_crit = torch.nn.CrossEntropyLoss()

        # sp loss, binary cross entropy
        sp_loss = bce_crit(torch.sigmoid(sp_logits), sp_label.float())

        return sp_loss


class GraphBasedModel(BertPreTrainedModel):

    def __init__(self, config, num_answer_type=3, num_hop=3, num_rel=2, no_gnn=False, edp=0.0,
                 sent_with_cls=False):
        super(GraphBasedModel, self).__init__(config)
        self.bert = BertModel(config)
        # self.model_freeze()

        self.dropout = config.hidden_dropout_prob
        self.num_rel = num_rel
        self.config = config
        self.sent_with_cls = sent_with_cls
        self.no_gnn = no_gnn

        # if not self.no_gnn:
        #     self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,edp=edp)

        self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,
                                 edp=edp)

        self.hidden_size = int(config.hidden_size / 2)

        self.ori_hidden = int(config.hidden_size)

        self.sent_selfatt = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(),
                                          nn.Dropout(self.dropout), nn.Linear(self.hidden_size, 1))

        self.sp_classifier = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(),
                                           nn.Dropout(self.dropout),
                                           nn.Linear(self.hidden_size, 1))  # input: graph embeddings

        self.num_answer_type = num_answer_type
        self.sfm = nn.Softmax(-1)
        # self.answer_type_classifier.half()

        self.init_weights()

    def attention(self, x, z):
        # x: batch_size X max_nodes X feat_dim
        # z: attention logits

        att = self.sfm(z).unsqueeze(-1)  # batch_size X max_nodes X 1

        output = torch.bmm(att.transpose(1, 2), x)

        return output

    def gen_mask(self, max_len, lengths, device):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num,
                                                                    -1) + 1  # +1 for masking out sequences with length 0
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask

    # self attentive pooling
    def do_selfatt(self, input, input_len, selfatt, span_logits=None):

        # input: max_len X batch_size X dim

        input_mask = self.gen_mask(input.size(0), input_len, input.device)

        att = selfatt.forward(input).squeeze(-1).transpose(0, 1)
        att = att.masked_fill(input_mask, -9e15)
        if span_logits is not None:
            att = att + span_logits
        att_sfm = self.sfm(att).unsqueeze(1)

        # print(att_sfm[56:63,:,:])
        # exit()

        output = torch.bmm(att_sfm, input.transpose(0, 1)).squeeze(1)  # batchsize x dim

        return output

    def forward(self, input_ids, input_mask, segment_ids, adj_matrix, graph_mask, sent_start,
                sent_end, sent_num=None ,sp_label=None, sent_sum_way='attn', gtem='ori'):

        """
        input_ids: bs X num_doc X num_sent X sent_len
        token_type_ids: same size as input_ids
        attention_mask: same size as input_ids
        input_adj_matrix: bs X 3 X max_nodes X max_nodes
        input_graph_mask: bs X max_nodes
        """

        # Roberta doesn't use token_type_ids cause there is no NSP task
        segment_ids = torch.zeros_like(segment_ids).to(input_ids.device)

        if gtem == 'ori':
            # reshaping
            bs, sent_len = input_ids.size()
            max_nodes = adj_matrix.size(-1)

            sequence_output, cls_output = self.bert(input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask)

            feat_dim = cls_output.size(-1)

            # sentence extraction
            per_sent_len = sent_end - sent_start
            max_sent_len = torch.max(sent_end - sent_start)
            if self.sent_with_cls:
                per_sent_len += 1
                max_sent_len += 1
            # print("Maximum sent length is {}".format(max_sent_len))
            sent_output = torch.zeros(bs, max_nodes, max_sent_len, feat_dim).to(input_ids.device)
            for i in range(bs):
                for j in range(max_nodes):
                    if sent_end[i, j] <= sent_len:
                        if sent_start[i, j] != -1 and sent_end[i, j] != -1:
                            if not self.sent_with_cls:
                                sent_output[i, j, :(sent_end[i, j] - sent_start[i, j]), :] = sequence_output[i,
                                                                                             sent_start[i, j]:sent_end[
                                                                                                 i, j], :]
                            else:
                                sent_output[i, j, 1:(sent_end[i, j] - sent_start[i, j]) + 1, :] = sequence_output[i,
                                                                                                  sent_start[i, j]:
                                                                                                  sent_end[
                                                                                                      i, j], :]
                                sent_output[i, j, 0, :] = cls_output[i]
                    else:
                        if sent_start[i, j] < sent_len:
                            if not self.sent_with_cls:
                                sent_output[i, j, :(sent_len - sent_start[i, j]), :] = sequence_output[i,
                                                                                       sent_start[i, j]:sent_len, :]
                            else:
                                sent_output[i, j, 1:(sent_len - sent_start[i, j]) + 1, :] = sequence_output[i,
                                                                                            sent_start[i, j]:sent_len,
                                                                                            :]
                                sent_output[i, j, 0, :] = cls_output[i]

            # sent summarization
            if sent_sum_way == 'avg':
                sent_sum_output = sent_output.mean(dim=2)
            elif sent_sum_way == 'attn':
                sent_sum_output = self.do_selfatt(
                    sent_output.contiguous().view(bs * max_nodes, max_sent_len, self.config.hidden_size).transpose(0,
                                                                                                                   1),
                    per_sent_len.view(bs * max_nodes), self.sent_selfatt).view(bs, max_nodes, -1)
        else:
            bs, sl, sent_len = input_ids.size()
            max_nodes = adj_matrix.size(-1)
            sent_sum_output = torch.zeros(bs, max_nodes, self.ori_hidden).to(input_ids.device)
            for i in range(bs):
                # print(sent_num[i])
                ll = int(sent_num[i])
                temp = ll
                while temp>50:
                    sent_sum_output[i, ll-temp:ll-temp+50, :] = self.bert(input_ids[i,ll-temp:ll-temp+50], token_type_ids=segment_ids[i,ll-temp:ll-temp+50],
                                                           attention_mask=input_mask[i, ll-temp:ll-temp+50])[1].squeeze(0)
                    temp -= 50
                sent_sum_output[i, ll-temp:ll, :] = self.bert(input_ids[i,:ll], token_type_ids=segment_ids[i,:ll],
                                                         attention_mask=input_mask[i,:ll])[1].squeeze(0)


        # graph reasoning
        if not self.no_gnn and self.num_rel > 0:
            gcn_output = self.sp_graph(sent_sum_output, graph_mask, adj_matrix)  # bs X max_nodes X feat_dim
        else:
            # gcn_output = self.sp_graph(sent_sum_output, graph_mask, torch.zeros(bs,1,max_nodes,max_nodes).to(input_ids.device))
            gcn_output = sent_sum_output

        # sp sent classification
        sp_logits = self.sp_classifier(gcn_output).view(bs, max_nodes)
        sp_logits = torch.where(graph_mask > 0, sp_logits, -9e15 * torch.ones_like(sp_logits).to(input_ids.device))

        # print(sp_logits)
        # select top 10 sentences with highest logits and then recalculate start and end logits

        # answer type logits

        if sp_label is not None:
            return self.loss_func(sp_logits, sp_label), sp_logits
        else:
            return sp_logits

    def loss_func(self, sp_logits, sp_label):

        bce_crit = torch.nn.BCELoss()
        ce_crit = torch.nn.CrossEntropyLoss()

        # sp loss, binary cross entropy
        sp_loss = bce_crit(torch.sigmoid(sp_logits), sp_label.float())

        return sp_loss
