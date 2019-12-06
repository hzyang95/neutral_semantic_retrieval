import logging

'''
6191
1173
'''
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data.dataloader as DataLoader
from allennlp.nn.util import move_to_device
from neu_sem_retrieval.model import Classifier
from neu_sem_retrieval.utils import parse_config, mkdata, subDataset, getData,getTestData,mktestdata
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

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
        self.train_data = getData(conf_file)
        self.test_data = getTestData(conf_file)
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
        if self.device_num >= 0:
            self.device = 'cuda'

    def train_retr(self):
        """
        Returns:
        """
        self.model.to(self.device)
        logger.info('begin training, model_file: {}, gpu: {}'.format(self.model_file, self.device_num))
        # print(tokenizer.tokenize('if sub_text not in self.added_tokens_encoder '))
        train, target_train = mkdata(self.tokenizer, self.train_data)
        # test, target_test, len_list_test = mkdata(self.tokenizer, self.test_data)
        # print(target_train)
        # print(input_ids[:10])
        # bertmodel = BertModel.from_pretrained(language).cuda()
        train = rnn_utils.pad_sequence(train, batch_first=True)  # , batch_first=True
        # test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
        dataset_train = subDataset(train, target_train)
        # dataset_test = subDataset(test, target_test, len_list_test)
        # print(dataset)
        tr = dataset_train.__len__()
        # intv = dataset_test.__len__()
        logger.info('train dataset大小为：{}'.format(tr))
        # logger.info('test dataset大小为：', intv)
        # print(dataset.__getitem__(0))
        # print(dataset[0])

        # 创建DataLoader迭代器
        dataloader_train = DataLoader.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        # dataloader_test = DataLoader.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # print(self.model.parameters())
        base_params = list(map(id, self.model.bert.parameters()))
        logits_params = filter(lambda p: id(p) not in base_params, self.model.parameters())
        # params = [
        #     {"params": logits_params, "lr": 0.01},
        #     {"params": self.model.bert.parameters(), "lr": 1e-5},
        # ]

        # param_optimizer = list(self.model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {"params": logits_params, "lr": 0.1},
            {"params": self.model.bert.parameters(), "lr": 1e-5},
        ]
        # optimizer = optim.SGD(params)
        # optimizer = AdamW(params, warmup=0.1)
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 100000)
        # softmax = nn.Softmax()
        sigmoid = nn.Sigmoid()
        criterion = nn.CrossEntropyLoss()
        ma = 0
        _losss=1000000
        for epoch in range(self.epoch):
            logger.info('-----' + str(epoch))
            acloss=torch.tensor(float(0))
            for i, item in enumerate(dataloader_train):
                item = move_to_device(item, self.device_num)
                # print('i:', i)
                data, label = item
                optimizer.zero_grad()  # zero the gradient buffers
                self.model.train()
                outputs = self.model(data, labels=label)
                loss, res = outputs[:2]
                acloss += loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if i % 100 == 0:
                    logger.info('-' + str(i) + ' ' + str(loss))

            logger.info(str(acloss))
            logger.info('eval..')
            pr,losss = self.eval_retr(self.model, self.test_data)
            logger.info(pr)
            logger.info(losss)
            if losss<_losss:
                _losss=losss
                save_dir = '../models/para_10000_500_warm_best'
                torch.save(self.model, save_dir)
                with open('best_10000_500_para.txt','w',encoding='utf-8') as ff:
                    ff.write(str(epoch))
                    ff.write('\n')
                    ff.write(str(losss))
                logger.info('best model saved to: {}'.format(self.model_file))
            if epoch % 100 == 0:
                save_dir = '../models/para_10000_500_warm_epoch_' + str(epoch) + '_loss_' + str(losss)
                torch.save(self.model, save_dir)
                logger.info('model saved to: {}'.format(save_dir))
        logger.info('end training')
        return self.model

    def eval_retr(self, cls, dataloader_test):
        losss = torch.tensor(float(0))
        rp = 0
        intv = 0
        r_t = []
        i_t = []
        with torch.no_grad():
            for i, item in enumerate(dataloader_test):
                test, target, tops = mktestdata(self.tokenizer, item)
                b_l = len(test)
                intv += b_l
                test = rnn_utils.pad_sequence(test, batch_first=True)  # , batch_first=True
                dataset_test = subDataset(test, target)
                _dataloader_test = DataLoader.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=4)

                cls.eval()
                res = []
                for id,it in enumerate(_dataloader_test):
                    it = move_to_device(it, self.device_num)
                    test,target_l = it
                    outputs = cls(test, labels=target_l)
                    loss, res_1 = outputs[:2]
                    sigmoid = nn.Sigmoid()
                    res_1 = sigmoid(res_1).tolist()
                    res += res_1
                    losss += loss
                res=torch.tensor(res)
                tops = torch.topk(res, 1, dim=0)
                ind = [0] * b_l
                for ii in tops[1]:
                    ind[ii[0]] = 1
                ind = torch.tensor(ind).to(self.device)
                # print(res)
                # print(target)
                # print(ind)
                r_t += target.tolist()
                i_t += ind.tolist()
                target=move_to_device(target,self.device_num)
                p = (ind == target).sum()
                rp += p.item()
                if i % 100 == 0:
                    logger.info('-' + str(i) + ' ' + str(loss))
        return (float(rp) / float(intv)),losss#, r_t, i_t

    # def eval(self, cls, dataloader_test, intv):
    #     rp = 0
    #     for i, item in enumerate(dataloader_test):
    #         # print('i:', i)
    #         item = move_to_device(item, self.device_num)
    #         data, label, len_ = item
    #         #
    #         # print('data:', data)
    #         # print('label:', label)
    #         # data = bertmodel(data.cuda())[0]
    #         cls.eval()
    #         outputs = cls(data, labels=label)
    #         loss, res = outputs[:2]
    #         # softmax = nn.Softmax()
    #         sigmoid = nn.Sigmoid()
    #         res = sigmoid(res)
    #         # print(res)
    #         # ind = torch.tensor([0 if i[0] < 0.5 else 1 for i in res])
    #         tops = torch.topk(res, int(self.batch_size / 2), dim=0)
    #         ind = [0] * self.batch_size
    #         for ii in tops[1]:
    #             ind[ii[0]] = 1
    #         ind = torch.tensor(ind).to(self.device)
    #         # print(out)
    #         # ind = torch.argmax(res, dim=1).to(self.device)
    #         # print('++++++++++')
    #         # print(label)
    #         # print(ind)
    #         p = (ind == label).sum()
    #         rp += p.item()
    #         if i % 20 == 0:
    #             logger.info(str(label) + str(ind))
    #             logger.info('-' + str(i) + ' ' + str(loss))
    #     return float(rp) / float(intv)