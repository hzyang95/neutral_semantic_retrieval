import logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from .eval_utils import cal_acc, count_label, cal_prf

def load_torch_model(model_path, use_cuda=True):
    with open(model_path + "/model.pt", "rb") as f:
        if use_cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
            model.cpu()
        model.eval()
        return model

def pickle_to_data(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'rb') as f:
        your_dict = pickle.load(f)
        return your_dict

def data_to_pickle(your_dict, out_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(out_file, 'wb') as f:
        pickle.dump(your_dict, f,protocol = 4)



def get_data(features):
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return data


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


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


def process_logit(batch_index, batch_logits,_sent_len):
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
        pred_sp_idx = [x[0] for x in enumerate(sp_logits_np[idx, :].tolist()) if x[1] > 0.5]
        # print(len(pred_sp_idx))
        # print(_sent_len[idx])

        top = 4
        # tops = torch.topk(torch.tensor(pred_sp_idx), min(top, _sent_len[idx]), dim=0)

        # if len(pred_sp_idx) != 0:
        #     sp_pred+=get_sp_pred(pred_sp_idx, predict_examples[idx])

        ind = [0] * _sent_len[idx]
        # for ii in tops[1]:
        #     ind[ii] = 1
        for ii in pred_sp_idx:
            ind[ii] = 1
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