import fasttext
import logging
from .utils import parse_config
from pytorch_transformers import BertTokenizer,BertModel
from typing import List
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
logger = logging.getLogger()
# file = logging.FileHandler(str(torch.cuda.device_count())+'gpulog',encoding='utf-8')
# file.setLevel(level=logging.INFO)
# logger.addHandler(file)

class Classifier(nn.Module):
    def __init__(self, conf_file):
        """
        Args:
            conf_file: 配置文件，包括训练数据、模型位置等文件路径配置以及模型参数配置。
        """
        super(Classifier,self).__init__()
        logger.info('parse config from: {}'.format(conf_file))
        conf = parse_config(conf_file)
        data_conf, params_conf = conf['path'], conf['params']
        self.language = 'english'
        if params_conf['choice'] == 1:
            self.language = 'chinese'
        self.embedding_dim = params_conf['embedding']
        self.hidden_dim = params_conf['hidden']
        self.res_dim = params_conf['res']
        # self.tokenizer = BertTokenizer.from_pretrained(params_conf[self.language])
        self.bertmodel = BertModel.from_pretrained(params_conf[self.language])
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim, batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim, self.res_dim)
        self.dropout = nn.Dropout(0.1)

        # self.l1 = nn.Linear(self.hidden_dim,500)
        # self.l2 = nn.Linear(500, 250)
        # self.l3 = nn.Linear(250, 125)
        # self.l4 = nn.Linear(125, self.res_dim)


    # input_ids = torch.tensor(tokenizer.encode("你好")).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]
    # print(last_hidden_states)

    def forward(self, input_ids):
        # input_ids = [torch.tensor(self.tokenizer.encode(i[0])) for i in tokens]
        # target = torch.tensor([i[1] for i in tokens])
        # # print(input_ids)
        # input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True) #, batch_first=True
        # print(input_ids)
        # outputs = self.bert_model(input_ids)
        # last_hidden_states = outputs[0]
        # last_hidden_states = input_ids
        # print(last_hidden_states)
        # print(last_hidden_states.size())
        # input_ids=rnn_utils.pack_padded_sequence(input_ids, len, batch_first=True,enforce_sorted=False)
        # aftlstm, (hn, cn)=self.lstm(input_ids)

        # print(hn.size())
        # print(hn)
        # print(cn.size())
        # print(aftlstm)
        # aftlstm,out_len = rnn_utils.pad_packed_sequence(aftlstm, batch_first=True)
        # print(aftlstm)
        # print(aftlstm[:,-1,:])
        # print(aftlstm[:, -1, :].size())
        # aftlinear = self.l1(hn[-1,:,:])
        # print(aftlinear.size())
        # print(target.size())
        # print(out)
        # print(loss)
        x = self.bertmodel(input_ids.cuda())[0]

        # print(x)
        # x = F.relu(self.l1(x[:,0,:]))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = self.l4(x)
        
        x = self.dropout(x)
        x = self.l1(x[:, 0, :])
        return x

