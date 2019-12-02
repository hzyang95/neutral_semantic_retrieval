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
from transformers import BertTokenizer,BertModel,BertForSequenceClassification,AdamW

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, conf_file):
        """
        Args:
            conf_file: 配置文件，包括训练数据、模型位置等文件路径配置以及模型参数配置。
        """
        logger.info('parse config from: {}'.format(conf_file))
        conf = parse_config(conf_file)
        data_conf, params_conf = conf['path'], conf['params']
        self.train_data,self.test_data = getData(conf_file)
        self.model_file = data_conf['model_file']
        self.epoch = params_conf['epoch']
        self.batch_size = params_conf['batch_size']
        self.learning_rate = params_conf['learning_rate']
        self.language = 'english'
        if params_conf['choice'] == 1:
            self.language = 'chinese'
        self.tokenizer = BertTokenizer.from_pretrained(params_conf[self.language])
        self.model = BertForSequenceClassification.from_pretrained(params_conf[self.language], num_labels=1)
        self.device_num = params_conf['gpu']
        self.device = 'cpu'
        if self.device_num>=0:
            self.device = 'cuda'
    def train(self):
        """
        Returns:
        """
        self.model.to(self.device)
        logger.info('begin training, model_file: {}, gpu: {}'.format(self.model_file,self.device_num))
        # print(tokenizer.tokenize('if sub_text not in self.added_tokens_encoder '))
        train, target_train, len_list_train = mkdata(self.tokenizer, self.train_data)
        test, target_test, len_list_test = mkdata(self.tokenizer, self.test_data)
        # print(target_train)
        # print(input_ids[:10])
        # bertmodel = BertModel.from_pretrained(language).cuda()
        train = rnn_utils.pad_sequence(train, batch_first=True)  # , batch_first=True
        test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
        dataset_train = subDataset(train, target_train, len_list_train)
        dataset_test = subDataset(test, target_test, len_list_test)
        # print(dataset)
        tr = dataset_train.__len__()
        intv = dataset_test.__len__()
        print('train dataset大小为：', tr)
        print('test dataset大小为：', intv)
        # print(dataset.__getitem__(0))
        # print(dataset[0])

        # 创建DataLoader迭代器
        dataloader_train = DataLoader.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        dataloader_test = DataLoader.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # print(self.model.parameters())
        # base_params = list(map(id, self.model.bert.parameters()))
        # logits_params = filter(lambda p: id(p) not in base_params, self.model.parameters())
        # params = [
        #     {"params": logits_params, "lr": 0.01},
        #     {"params": self.model.bert.parameters(), "lr": 1e-5},
        # ]

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # optimizer = optim.SGD(params)
        optimizer = AdamW(optimizer_grouped_parameters,
                             lr=5e-5)
        softmax = nn.Softmax()
        sigmoid = nn.Sigmoid()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epoch):
            print('-----' + str(epoch))
            rp = 0
            for i, item in enumerate(dataloader_train):
                item = move_to_device(item,self.device_num)
                # print('i:', i)
                data, label, len_ = item
                #
                # print('data:', data)
                # print('label:', label)

                # print('dataaftbert:', data.shape)
                # data = rnn_utils.pack_padded_sequence(data, len, batch_first=True,enforce_sorted=False)
                optimizer.zero_grad()  # zero the gradient buffers
                self.model.train()
                # print(data.size())
                # print(label.size())
                outputs = self.model(data, labels=label)
                loss, res = outputs[:2]
                # print(outputs)
                # print(res)
                res = sigmoid(res)
                # print(res)
                # print(res.size())
                # print(softmax(res))
                # print('---------')
                # print(loss)
                loss.backward()
                optimizer.step()
                # ind  = torch.tensor([0 if i[0]<0.5 else 1 for i in res])
                topk = torch.topk(res, int(self.batch_size/2), dim=0)
                ind = [0]*self.batch_size
                for ii in topk[1]:
                    ind[ii[0]] = 1
                ind = torch.tensor(ind).to(self.device)
                # # ind = torch.argmax(res, dim=1).to(self.device)
                # print(label)
                # print(ind)
                p = (ind == label).sum()
                rp += p.item()
                if i % 20 == 0:
                    print(str(label)+str(ind))
                    print('-' + str(i) + ' ' + str(loss))
            print(float(rp) / tr)
            rp = 0
            print('eval..')
            pr=self.eval(self.model,dataloader_test)
            print(pr)
            if (epoch%20==0):
                save_dir='../models/new_epoch_'+str(epoch)+'_pr_'+str(pr)
                torch.save(self.model,save_dir)
                logger.info('model saved to: {}'.format(self.model_file))
        logger.info('end training')
        return self.model

    def eval(self,cls, dataloader_test):
        rp = 0
        for i, item in enumerate(dataloader_test):
            # print('i:', i)
            item = move_to_device(item, self.device_num)
            data, label, len_ = item
            #
            # print('data:', data)
            # print('label:', label)
            # data = bertmodel(data.cuda())[0]
            cls.eval()
            outputs = self.model(data, label, len_)
            loss, res = outputs[:2]
            softmax = nn.Softmax()
            sigmoid = nn.Sigmoid()
            res = sigmoid(res)
            # print(res)
            # ind = torch.tensor([0 if i[0] < 0.5 else 1 for i in res])
            topk = torch.topk(res, self.batch_size / 2, dim=0)
            ind = [0] * self.batch_size
            for ii in topk[1]:
                ind[ii[0]] = 1
            ind = torch.tensor(ind).to(self.device)
            # print(out)
            # ind = torch.argmax(res, dim=1).to(self.device)
            # print('++++++++++')
            # print(label)
            # print(ind)
            p = (ind == label).sum()
            rp += p.item()
            if i % 20 == 0:
                print(str(label)+str(ind))
                print('-' + str(i) + ' ' + str(loss))
        return (float(rp) / (-1 * intv))