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
from neu_sem_retrieval.utils import parse_config,mkdata,subDataset,getData
# from data.getdata import getData
import numpy as np

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))
language='bert-base-uncased'
# tokens = [['i am fine',1],['hello bro',1],['shit',0],['son of bitch',0],['very good',1],['sounds good',1],['holly shit',0]]
# tokens = [['你好',1],['你特别好',1],['傻逼',0],['傻蛋',0],['厉害',1],['厉害',1],['傻吊',0]]
from pytorch_transformers import BertTokenizer,BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_num = 0 if torch.cuda.is_available() else -1


logger = logging.getLogger(__name__)




def eval(cls, dataloader_test, intv):
    rp = 0
    for i, item in enumerate(dataloader_test):
        move_to_device(item, device_num)
        # print('i:', i)
        data, label, len_ = item
        #
        # print('data:', data)
        # print('label:', label)
        # data = bertmodel(data.cuda())[0]
        cls.eval()
        res = cls(data, label, len_)
        softmax = nn.Softmax()
        criterion = nn.CrossEntropyLoss()
        out = softmax(res)
        loss = criterion(res, label)

        # print(out)
        ind = torch.argmax(out, dim=1)
        # print('++++++++++')
        # print(label)
        # print(ind)
        p = (ind == label).sum()
        rp += p.item()
        if i % 20 == 0:
            print('-' + str(i) + ' ' + str(loss))
    return (float(rp) / (-1 * intv))


topk = torch.topk(torch.tensor([[0.7829],[0.2532],[0.6864],[0.3352]]), 2,dim=0)
print(topk[0])
print(topk[1])

ind = [0]*4
for i in topk[1]:
    print(i[0])
    ind[i[0]]=1
print(ind)


# temp = torch.tensor([0 if i[0]<0.5 else 1 for i in torch.tensor([[0.7829],[0.2532],[0.6864],[0.3352]])])
# print(temp)

# if __name__ == '__main__':
#     save_dir='../models/two_epoch_80_pr_-0.5585'
#     cls = torch.load(save_dir)
#     cls.to(device)
#
#     conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
#     conf = parse_config(conf_file)
#     data_conf, params_conf = conf['path'], conf['params']
#     train_data,test_data = getData(conf_file)
#     model_file = data_conf['model_file']
#     epoch = params_conf['epoch']
#     batch_size = params_conf['batch_size']
#     language = 'english'
#     if params_conf['choice'] == 1:
#         language = 'chinese'
#     tokenizer = BertTokenizer.from_pretrained(params_conf[language])
#     test, target_test, len_list_test = mkdata(tokenizer, test_data)
#     # print(target_train)
#     # print(input_ids[:10])
#     # bertmodel = BertModel.from_pretrained(language).cuda()
#     # train = rnn_utils.pad_sequence(train, batch_first=True)  # , batch_first=True
#     test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
#     # dataset_train = subDataset(train, target_train, len_list_train)
#     dataset_test = subDataset(test, target_test, len_list_test)
#     dataloader_test = DataLoader.DataLoader(dataset_test, batch_size=6, shuffle=False, num_workers=4)
#
#     intv = dataset_test.__len__()
#     print('test dataset大小为：', intv)
#     pr=eval(cls,dataloader_test,intv)