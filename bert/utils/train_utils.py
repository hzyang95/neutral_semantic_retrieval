import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from .eval_utils import cal_acc, count_label, cal_prf
from .model_utils import load_torch_model


def pickle_to_data(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'rb') as f:
        your_dict = pickle.load(f)
        return your_dict


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
    print("-------------------------------")
    accuracy = cal_acc(y_pred, y_true)
    # for All kinds of classes
    pred, right, gold = count_label(y_pred, y_true, include_class=[0, 1, 2])
    prf_result = cal_prf(pred, right, gold, formation=False)
    p = prf_result['p']
    r = prf_result['r']
    f1 = prf_result['f']
    macro_f1 = prf_result["macro"][-1]
    micro_f1 = prf_result["micro"][-1]

    print("  *** Cons|Neu|Pros ***\n  ***", pred, right, gold)
    print("   *Accuracy is %d/%d = %f" % (sum(right), sum(gold), accuracy))
    print("    Precision: %s" % p)
    print("    Recall   : %s" % r)
    print("    F1 score : %s" % f1)
    print("    Macro F1 score on is %f" % macro_f1)
    print("    Micro F1 score on is %f" % micro_f1)

    # for classes of interest
    # pred, right, gold = count_label(y_pred, y_true, include_class=[0, 2])
    # prf_result = cal_prf(pred, right, gold, formation=False)
    # p = prf_result['p']
    # r = prf_result['r']
    # f1 = prf_result['f']
    # macro_f1 = prf_result["macro"][-1]
    # micro_f1 = prf_result["micro"][-1]
    #
    # print("  *** Cons|Pros ***\n  ***", pred, right, gold)
    # print("   *Right on test is %d/%d = %f" % (sum(right), sum(gold), sum(right) / sum(gold)))
    # print("    Precision: %s" % p)
    # print("    Recall   : %s" % r)
    # print("    F1 score : %s" % f1)
    # print("    Macro F1 score on is %f" % macro_f1)
    # print("    Micro F1 score on is %f" % micro_f1)

    # eval_result = [accuracy, macro_f1, micro_f1]
    eval_result = {
        "accuracy": accuracy,
        "macro_f": macro_f1,
        "micro_f": micro_f1,
        "f_score": f1
    }

    return macro_f1  # [accuracy, f1{Con/Pro}, macro_f1]
