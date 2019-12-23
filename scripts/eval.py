import fasttext
import logging
import torch.optim as optim
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data.dataloader as DataLoader
from allennlp.nn.util import move_to_device

# from neu_sem_retrieval.model import Classifier
from neu_sem_retrieval.utils import parse_config, mkdata, subDataset, getData, getTestData, mktestdata

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from pytorch_transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_num = 0 if torch.cuda.is_available() else -1


# device = "cpu"
# device_num = -1


def eval(cls, dataloader_test, tokenizer):
    rp = 0
    intv = 0

    r_t = []
    i_t = []

    with torch.no_grad():
        cls.eval()
        for i, item in enumerate(dataloader_test):
            test, target, tops = mktestdata(tokenizer, item)
            b_l = len(test)
            intv += b_l
            test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
            test = test.to(device)
            target = target.to(device)

            outputs = cls(test, labels=target)
            loss, res = outputs[:2]

            # res_1 = self.model(test)
            # criterion = nn.MSELoss()
            # loss = criterion(res_1.view(-1,self.num_labels), target.view(-1))
            loss = torch.mean(loss)
            sigmoid = nn.Sigmoid()
            res = sigmoid(res)
            tops = torch.topk(res, min(1, b_l), dim=0)
            ind = [0] * b_l
            for ii in tops[1]:
                ind[ii[0]] = 1
            ind = torch.tensor(ind).to(device)
            # print(res)
            # print(target)
            # print(ind)
            r_t += target.tolist()
            i_t += ind.tolist()
            # target=move_to_device(target,self.device_num)
            target = target.to(device)
            p = (ind == target).sum()
            rp += p.item()
            if i % 100 == 0:
                print('-' + str(i) + ' ' + str(loss))
    return (float(rp) / float(intv)), r_t, i_t


# topk = torch.topk(torch.tensor([[0.7829],[0.2532],[0.6864],[0.3352]]), 2,dim=0)
# print(topk[0])
# print(topk[1])
#
# ind = [0]*4
# for i in topk[1]:
#     print(i[0])
#     ind[i[0]]=1
# print(ind)


# temp = torch.tensor([0 if i[0]<0.5 else 1 for i in torch.tensor([[0.7829],[0.2532],[0.6864],[0.3352]])])
# print(temp)

if __name__ == '__main__':
    # save_dir = '../models/new_epoch_60_pr_0.5045'
    # save_dir = '../models/new_epoch_40_pr_0.4935'
    save_dir = '../models/para_10000_200_warm_best_4gpu'
    # save_dir = '../models/para_10000_500_warm_best'
    # save_dir = '../models/para_20000_200_warm_best_4gpu'
    # save_dir = '../models/sent_53019_441_warm_best_4gpu'
    # save_dir = '../models/para_50000_500_warm_best'
    # save_dir = '../models/sent_53019_441_warm_best'

    cls = torch.load(save_dir).module
    cls.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(cls, [i for i in range(torch.cuda.device_count())])

    conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
    conf = parse_config(conf_file)
    data_conf, params_conf = conf['path'], conf['params']
    # train_data,test_data = getData(conf_file)
    test_data = getTestData(conf_file)
    model_file = data_conf['model_file']
    batch_size = params_conf['batch_size']
    language = 'english'
    if params_conf['choice'] == 1:
        language = 'chinese'
    tokenizer = BertTokenizer.from_pretrained(params_conf[language])

    pr, ref, res = eval(cls, test_data, tokenizer)

    print(pr)

    assert len(ref) == len(res)

    length = len(ref)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(length):
        a = ref[i]
        b = res[i]
        if a == 1:
            if b == 1:
                tp += 1
            else:
                fn += 1
        else:
            if b == 1:
                fp += 1
            else:
                tn += 1

    print("tp: " + str(tp))
    print("fp: " + str(fp))
    print("tn: " + str(tn))
    print("fn: " + str(fn))
    print("acc: " + str((tp + tn) / (tp + fp + tn + fn)))
    pr = tp / (tp + fp)
    print("pre: " + str(pr))
    rec = tp / (tp + fn)
    print("rec: " + str(rec))
    f1 = 2 * pr * rec / (pr + rec)
    print("f1:" + str(f1))
