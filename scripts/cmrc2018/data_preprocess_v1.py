import json
import re
import os
import random
import time
import jieba
from tqdm import tqdm
from pyltp import SentenceSplitter

_stage = ['train', 'dev']
stage = _stage[1]

_split = 'valid'

file_path = 'squad-style-data/cmrc2018_' + stage + '.json'

f = open(file_path, 'r', encoding='utf-8')
file = json.loads(f.read())['data']

print(len(file))
# first_num = 2000
data = []
_all = 0
_sel = 0
sent_len = 0
sent_num = 0
_len = 0
pas_len = 0
otp = 0

for line in file[:]:
    paras = line['paragraphs'][0]
    passage = paras['context'].strip().replace(' ', '').replace('\n', '').replace('\t', '')
    pas_len += len(passage)
    passage_sent = SentenceSplitter.split(passage)
    index_list = [0]
    for i, sent in enumerate(passage_sent):
        index_list += [i] * len(sent)
    # print(len(index_list))
    # print(index_list)

    qases = paras['qas']
    for qas in qases:
        qid = qas['id']
        query = qas['question'].strip().replace(' ', '').replace('\n', '').replace('\t', '')
        answers = qas['answers']
        _ans_list = []
        for answer in answers:
            answer_text = answer['text'].strip().replace(' ', '').replace('\n', '').replace('\t', '')
            answer_start = passage.find(answer_text)
            _ans_list += set(index_list[answer_start:answer_start + len(answer_text)])
        ans_list = list(set(_ans_list))
        # _all += len(passage_sent)
        # _sel += len(ans_list)
        # print(ans_list)
        temp_data = {'question': query, 'sample': [], 'top': len(ans_list)}
        for _ind, sent in enumerate(passage_sent):
            sent_len += len(sent)
            sent_num += 1
            if _ind in ans_list:
                label = 1
            else:
                label = 0
            # print(_ind,' ',label)
            if label == 0:
                _all += 1
                temp_data['sample'].append({"label": label, 'question': query, 'text': sent})
            else:
                for i in range(1):
                    _sel += 1
                    _all += 1
                    temp_data['sample'].append({"label": label, 'question': query, 'text': sent})
        data.append(temp_data)

#             if _split == 'valid':
#                 temp_data = {'question': query, 'sample': [], 'top': temp_num}
#                 for _ind, sent in enumerate(passage_sent):
#                     sent_len += len(sent)
#                     sent_num += 1
#                     temp_data['sample'].append({"label": bool_select[_ind], 'question': query, 'text': sent})
#                 data.append(temp_data)
#                 _len += 1
#             else:
#                 _len += 1
#                 for _ind, sent in enumerate(passage_sent):
#                     sent_len += len(sent)
#                     sent_num += 1
#                     data.append({"label": bool_select[_ind], 'question': query, 'text': sent})
_len = len(data)
aver_sel_sum = float(_sel) / float(_len)
aver_all_sum = float(_all) / float(_len)
aver_len = (float(sent_len) / float(sent_num))
print(_len)
print(_all)
print(_sel)
print(aver_sel_sum)
print(aver_all_sum)
print(aver_len)
print(pas_len/float(len(file)))

if otp:
    save_path = 'cmrc2018__' + stage + '.' + str(len(data)) + '.' + str(int(aver_sel_sum)) + '.' + str(
        int(aver_all_sum)) + '.' + str(aver_len) + '.json'
    with open(save_path, 'w', encoding='utf-8') as json_file:
        for each_dict in data:
            json_file.write(json.dumps(each_dict) + '\n')
