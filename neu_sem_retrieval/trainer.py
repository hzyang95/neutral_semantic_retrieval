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
        self.language = 'english'
        if params_conf['choice'] == 1:
            self.language = 'chinese'
        self.tokenizer = BertTokenizer.from_pretrained(params_conf[self.language])

    def train(self):
        """
        Returns:
        """
        logger.info('begin training, model_file: {}'.format(self.model_file))
        cls = Classifier(os.path.join(CUR_PATH, '../conf/config.yaml'))
        cls = cls.cuda()

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
        print(cls.parameters())
        base_params = list(map(id, cls.bertmodel.parameters()))
        logits_params = filter(lambda p: id(p) not in base_params, cls.parameters())
        params = [
            {"params": logits_params, "lr": 0.01},
            {"params": cls.bertmodel.parameters(), "lr": 1e-5},
        ]
        optimizer = optim.SGD(params)
        softmax = nn.Softmax()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epoch):
            print('-----' + str(epoch))
            rp = 0
            for i, item in enumerate(dataloader_train):
                # print('i:', i)
                data, label, len_ = item
                #
                # print('data:', data)
                # print('label:', label)

                # print('dataaftbert:', data.shape)
                # data = rnn_utils.pack_padded_sequence(data, len, batch_first=True,enforce_sorted=False)
                optimizer.zero_grad()  # zero the gradient buffers
                cls.train()
                # print(data.size())
                # print(label.size())
                res = cls(data.cuda(), label.cuda(), len_)
                # print(res)
                # print(res.size())
                # print(softmax(res))
                out = softmax(res)

                loss = criterion(res, label.cuda())
                # print('---------')
                # print(loss)
                loss.backward()
                optimizer.step()

                ind = torch.argmax(out, dim=1)
                # print(label)
                # print(ind)
                p = (ind.cuda() == label.cuda()).sum()
                rp += p.item()
                if i % 20 == 0:
                    print('-' + str(i) + ' ' + str(loss))
            print(float(rp) / tr)
            rp = 0
            print('eval..')
            pr=self.eval(cls,dataloader_test,intv)
            print(pr)
            if (epoch%20==0):
                save_dir='../models/two_epoch_'+str(epoch)+'_pr_'+str(pr)
                torch.save(cls,save_dir)
                logger.info('model saved to: {}'.format(self.model_file))
        logger.info('end training')
        return cls

    def eval(self,cls, dataloader_test, intv):
        rp = 0
        for i, item in enumerate(dataloader_test):
            # print('i:', i)
            data, label, len_ = item
            #
            # print('data:', data)
            # print('label:', label)
            # data = bertmodel(data.cuda())[0]
            cls.eval()
            res = cls(data.cuda(), label.cuda(), len_)
            softmax = nn.Softmax()
            criterion = nn.CrossEntropyLoss()
            out = softmax(res)
            loss = criterion(res, label.cuda())

            # print(out)
            ind = torch.argmax(out, dim=1)
            # print('++++++++++')
            # print(label)
            # print(ind)
            p = (ind.cuda() == label.cuda()).sum()
            rp += p.item()
            if i % 20 == 0:
                print('-' + str(i) + ' ' + str(loss))
        return (float(rp) / (-1 * intv))
    def predict(self, text):
        """
        Args:
            text: 待预测的文本或文本列表
        Returns:
            预测的类别或类别列表
        """
        logger.info('model load from : {}'.format(self.model_file))
        model = fasttext.load_model('{}.bin'.format(self.model_file))
