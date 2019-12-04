import fasttext
import logging

'''
6191
1173
'''
import torch.optim as optim
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data.dataloader as DataLoader
from allennlp.nn.util import move_to_device

from neu_sem_retrieval.model import Classifier
from neu_sem_retrieval.utils import parse_config, mkdata, subDataset, getData, getTestData, mktestdata

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from pytorch_transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device_num = 0 if torch.cuda.is_available() else -1
device = "cpu"
device_num = -1

logger = logging.getLogger(__name__)


def eval(cls, dataloader_test, tokenizer):
    rp = 0
    intv = 0

    r_t = []
    i_t = []
    for i, item in enumerate(dataloader_test):
        # item = move_to_device(item, device_num)
        # print('i:', i)
        # data, label, len_ = item
        #
        # print('data:', data)
        # print('label:', label)
        # data = bertmodel(data.cuda())[0]
        test, target, tops = mktestdata(tokenizer, item)
        # print(target_train)
        # print(input_ids[:10])
        # bertmodel = BertModel.from_pretrained(language).cuda()
        # train = rnn_utils.pad_sequence(train, batch_first=True)  # , batch_first=True
        b_l = len(test)
        intv += b_l

        test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
        # dataset_train = subDataset(train, target_train, len_list_train)

        # test.to(device)
        # target.to(device)

        test = move_to_device(test, device_num)
        target = move_to_device(target, device_num)

        cls.eval()

        outputs = cls(test, labels=target)
        loss, res = outputs[:2]
        # print(outputs)
        print(res)
        sigmoid = nn.Sigmoid()
        res = sigmoid(res)

        # res = cls(test, target, tops)
        # softmax = nn.Softmax()
        # criterion = nn.CrossEntropyLoss()
        # out = softmax(res)
        # loss = criterion(res, target)

        # print(out)
        # ind = torch.argmax(out, dim=1)
        # print('++++++++++')
        # min(2, b_l)
        tops = torch.topk(res,min(2, b_l) , dim=0)
        ind = [0] * b_l
        for ii in tops[1]:
            ind[ii[0]] = 1
        ind = torch.tensor(ind).to(device)
        print(res)
        print(target)
        print(ind)
        r_t += target.tolist()
        i_t += ind.tolist()
        p = (ind == target).sum()
        rp += p.item()
        if i % 20 == 0:
            print('-' + str(i) + ' ' + str(loss))
    return (float(rp) / (-1 * intv)), r_t, i_t


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
    # save_dir = '../models/new_epoch_80_pr_0.4935'
    # save_dir = '../models/zhidao_new_epoch_20_pr_-0.507057546145494'
    save_dir = '../models/doc_5000_epoch_140_pr_-0.5200868621064061'

    cls = torch.load(save_dir)
    cls.to(device)

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
        if a==1:
            if b==1:
                tp += 1
            else:
                fn += 1
        else:
            if b==1:
                fp += 1
            else:
                tn += 1

    print("tp: "+str(tp))
    print("fp: " + str(fp))
    print("tn: " + str(tn))
    print("fn: " + str(fn))
    print("acc: " + str((tp+tn) / (tp + fp +tn +fn)))
    print("pre: "+ str(tp/(tp+fp)))
    print("rec: " + str(tp / (tp + fn)))