import logging
from collections import Counter
import numpy as np
import random
import time

import torch

random.seed(1)
np.random.seed(1234567)


#############################################################
def cal_acc(y_pred, y_true):
    y_pred = np.asarray([int(i) for i in y_pred])
    y_true = np.asarray([int(i) for i in y_true])

    right = np.count_nonzero(y_pred == y_true)
    total = len(y_true)

    return 1.0 * right / total


def cal_prf(pred, right, gold, formation=True):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_label = len(pred)
    precisions = np.zeros(num_label)
    recalls = np.zeros(num_label)
    f_scores = np.zeros(num_label)

    for i in range(num_label):
        ''' cal precision for each class: right / predict '''
        precisions[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recalls[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f_scores[i] = 0 if precisions[i] == 0 or recalls[i] == 0 \
            else 2.0 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

        if formation:
            precisions[i] = precisions[i].__format__(".6f")
            recalls[i] = recalls[i].__format__(".6f")
            f_scores[i] = f_scores[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    prf_result = dict()

    prf_result["p"] = precisions
    prf_result['r'] = recalls
    prf_result["f"] = f_scores

    macro_p = sum(precisions) / num_label
    macro_r = sum(recalls) / num_label
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0

    micro_p = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
    micro_r = 1.0 * sum(right) / sum(gold) if sum(gold) > 0 else 0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    # macro_p, macro_r, macro_f = _cal_macro_f1(precisions, recalls, num_label)
    # micro_p, micro_r, micro_f = _cal_micro_f1(pred, right, gold)
    prf_result["macro"] = [macro_p, macro_r, macro_f]
    prf_result["micro"] = [micro_p, micro_r, micro_f]

    return prf_result


def _cal_macro_f1(precisions, recalls, num_label):
    precision = sum(precisions) / num_label
    recall = sum(recalls) / num_label
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f_score


def _cal_micro_f1(pred, right, gold):
    precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
    recall = 1.0 * sum(right) / sum(gold) if sum(gold) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f_score


def gen_confusion_matrix(y_pred, y_true, include_class=None):
    """
    Using numpy instead of sk-learn for generating confusion matrix
    :param y_pred:
    :param y_true:
    :param include_class:
    :return:
    """
    y_pred = [int(i) for i in y_pred]
    y_true = [int(i) for i in y_true]
    cnt = Counter(y_true)
    num_class = len(cnt)

    cm = np.zeros((num_class, num_class), dtype=np.int32)
    for idx in range(len(y_true)):
        # y_pred * y_true
        cm[y_true[idx]][y_pred[idx]] += 1

    if include_class is not None:
        # print(c[[0, 2]][:, [0, 2]])
        cm = cm[include_class][:, include_class]
    return cm


def count_label(y_pred, y_true, include_class=None):
    """
    Given the y_pred and gold labels, count the prediction/right/gold array
    :param y_pred: prediction results
    :param y_true: gold labels
    :param include_class: labels included
    :return:
    """
    cm = gen_confusion_matrix(y_pred, y_true)
    num_class = cm.shape[0]

    pred = np.sum(cm, axis=0)
    right = np.take(cm, indices=[i + i * num_class for i in range(num_class)])
    gold = np.sum(cm, axis=1)

    # include_class should be processed here!
    if include_class is not None:
        pred = [pred[i] for i in range(num_class) if i in include_class]
        right = [right[i] for i in range(num_class) if i in include_class]
        gold = [gold[i] for i in range(num_class) if i in include_class]

    return pred, right, gold


def log_prf_single(y_pred, y_true):
    """
    cal prf and macro-f1 for single model
    :param y_true:
    :param y_pred:
    :param model_name:
    :return:
    """
    logging.info("-------------------------------")
    accuracy = cal_acc(y_pred, y_true)
    # for All kinds of classes
    pred, right, gold = count_label(y_pred, y_true, include_class=[0, 1, 2])
    prf_result = cal_prf(pred, right, gold, formation=False)
    p = prf_result['p']
    r = prf_result['r']
    f1 = prf_result['f']
    macro_f1 = prf_result["macro"][-1]
    micro_f1 = prf_result["micro"][-1]

    logging.info("  *** Cons|Neu|Pros ***")
    logging.info("  *** {} {} {} ***".format(pred, right, gold))
    logging.info("    Accuracy is %d/%d = %f" % (sum(right), sum(gold), accuracy))
    logging.info("    Precision: %s" % p)
    logging.info("    Recall   : %s" % r)
    logging.info("    F1 score : %s" % f1)
    logging.info("    Macro F1 score on is %f" % macro_f1)
    logging.info("    Micro F1 score on is %f" % micro_f1)

    return macro_f1


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

    sp_pred, span_pred, ans_type_pred = [], [], []

    for idx, data in enumerate(batch_index):
        # print(sp_logits_np)
        # supporting facts prediction
        # pred_sp_idx = [x[0] for x in enumerate(sp_logits_np[idx, :].tolist()) if x[1] > 0]
        for x in enumerate(sp_logits_np[idx, :].tolist()):
            pred_sp_idx = [x[0]]
        # print(len(pred_sp_idx))
        # print(_sent_len)
        # print(_sent_len[idx])
 
        top = 4
        tops = torch.topk(torch.tensor(sp_logits_np[idx, :]), min(top, _sent_len[idx]), dim=0)
        # print(tops)
        # if len(pred_sp_idx) != 0:
        #     sp_pred+=get_sp_pred(pred_sp_idx, predict_examples[idx])

        ind = [0] * _sent_len[idx]
        for ii in tops[1]:
            ind[ii] = 1
        # for ii in pred_sp_idx:
        #     ind[ii] = 1
        # sp_pred += pred_sp_idx
        sp_pred += ind

    return sp_pred


def get_sp_pred(pred_sp_idx, data):
    """get the prediction of supporting facts in original format

    Arguments:
        pred_sp_idx {[type]} -- [description]
        data {[type]} -- [description]
    """
    pred = []
    for p in pred_sp_idx:
        if p < len(data):
            pred.append([data[p].doc_title[0], data[p].sent_id])

    return pred


if __name__ == "__main__":
    print("------------This is for test--------------")
    dt = np.random.rand(4, 5)
    xlabels = ["x" + str(i) for i in range(4)]
    ylabels = ["y" + str(i) for i in range(5)]

