import json
import logging
import os

from tqdm import tqdm
from transformers import BertTokenizer

# from graph.utils.ner_and_np import get_NER, get_NER_LTP, get_NP
# from graph.utils.tools import data_to_pickle, pickle_to_data
from utils.train_utils import InputExample, convert_examples_to_features_sent_ques

logger = logging.getLogger(__name__)


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


class DataProcessorFull(object):

    def get_labels(self):
        return [False, True]

    def create_examples(self, path, set_type, test=False, is_train=True, sent=False):
        # print(path+'.pkl')
        # print(os.path.exists(path+'.pkl'))
        # if os.path.exists(path+'.pkl'):
        #     print('get!'+path)
        #     return pickle_to_data(path+'.pkl')
        examples = []
        print(sent)
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        if test:
            file = file[:200]
        print(len(file))
        _aver = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        nnnn = 0
        neg = []
        neg_long = []
        all_num = 0
        # lb0 = 0
        ex = []
        for i, line in enumerate(tqdm(file[:])):
            _row = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            question = _row['question']
            # ques_np_ety = get_NER(question) + get_NER_LTP(question) + get_NP(question)
            # ques_np_ety = list(set(ques_np_ety))
            ques_np_ety = _row['q_ner'] + _row['q_ner_ltp'] + _row['q_np']
            ques_np_ety = list(set(ques_np_ety))
            content = []
            label = []
            ll = 0
            # if row['top'] == 0:
            #     lb0 += 1
            for doc_id, row in enumerate(_row['examples']):
                for item in row['sample']:
                    lll = len(item['text'])
                    # print(ll)
                    if check_contain_chinese(item['text']) is False:
                        continue
                    # if lll // 100 < 4:
                    #     _aver[lll // 100] += 1
                    # else:
                    #     _aver[4] += 1
                    #     # print(lll)
                    #     # print(item['text'])
                    if lll < 5:
                        neg.append(question + ':    ' + item['text'])
                        continue
                    if lll > 200:
                        # if lll < 300:
                        neg_long.append(question + ':   ' + item['text'])
                        continue
                    # print(item['text'])
                    # print(get_NER(item['text']))
                    # print(get_NER_LTP(item['text']))
                    # print(get_NP(item['text']))
                    # ans_np_ety = get_NER(item['text']) + get_NER_LTP(item['text']) + get_NP(item['text'])
                    # ans_np_ety = list(set(ans_np_ety))
                    ans_np_ety = item['a_ner'] + item['a_ner_ltp'] + item['a_np']
                    ans_np_ety = list(set(ans_np_ety))
                    # print(ans_np_ety)
                    if (not sent and (ll + lll > 400 or len(content) + 1 >= 30)) or (
                            sent and (len(content) + 1 >= 100)):
                        # if ll + lll > 400 or len(content) + 1 >= 30:
                        # print(len(content))
                        if len(label) == 0:
                            continue
                        assert len(content) == len(label)
                        # ssss = 0
                        # for i in label:
                        #     if i == 1:
                        #         ssss = 1
                        #         break
                        # if ssss == 0:
                        #     nnnn += 1
                        len_of_cont = len(content)
                        _aver[len_of_cont // 10] += 1
                        # if lll // 10 < 3:
                        #     _aver[lll // 10] += 1
                        # else:
                        #     _aver[3] += 1
                        #     # print(lll)
                        #     # print(content)
                        #     continue
                        all_num += len_of_cont
                        examples.append(InputExample(guid=guid, question=question, answer=content, label=label))
                        content = []
                        label = []
                        ll = 0
                    ll += lll
                    content.append({'text': item['text'], 'ans_np_ety': ans_np_ety, 'doc_id': doc_id})
                    label.append(item['label'])
            # print(len(content))
            if len(label) == 0:
                continue
            assert len(content) == len(label)
            len_of_cont = len(content)
            if len_of_cont // 10 < 10:
                _aver[len_of_cont // 10] += 1
            else:
                _aver[10] += 1
                ex.append(len_of_cont)
            all_num += len_of_cont
            # if lll // 10 < 3:
            #     _aver[lll // 10] += 1
            # else:
            #     _aver[3] += 1
            #     continue
            #     # print(ll)
            #     # print(content)
            # if ll // 100 < 5:
            #     _aver[ll // 100] += 1
            # else:
            #     _aver[5] += 1
            #     continue
            # ssss = 0
            # for i in label:
            #     if i == 1:
            #         ssss = 1
            #         break
            # if ssss == 0:
            #     nnnn += 1
            # print(question)
            # print(content)
            # print(row['sample'])
            # print(row['top'])
            # print(label)
            # if len('.'.join(content))>300:
            #     continue
            # if ll > 300:
            #     continue
            assert len(label) == len(content)
            examples.append(
                InputExample(guid=guid, question=question, ques_np_ety=ques_np_ety, answer=content, label=label))
        print(_aver)
        print(ex)
        print(all_num)
        # print(len(neg))
        # # print('\n'.join(neg[:10]))
        # print(len(neg_long))
        # # print('\n'.join(neg_long[:10]))
        # # print(nnnn)
        # # print(lb0)
        print(len(examples))
        logging.info(str(set_type) + str(len(examples)))
        # data_to_pickle(examples, path + ".pkl")
        return examples


if __name__ == "__main__":
    dataset = [
        # 'data_for_graph_train.v1.30000.143127.2.8.41.json',
        # 'data_for_graph_valid.2000.10115.3.7.41.json',
        # 'data_for_graph_test.3000.13848.3.9.40.json',
        'ques_data_for_graph_train.v1.30000.27127.2.8.41.json',
        'ques_data_for_graph_valid.2000.10115.1809.3.7.41.json',
        'ques_data_for_graph_test.3000.13848.2621.3.9.40.json'
    ]
    dt = DataProcessorFull()
    tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')
    for i in dataset:
        fe = dt.create_examples('../data/' + i+'.ety', 'test', False, True, sent=True)
        # data_to_pickle(fe, "data/" + i[:-5] + ".pkl")
        # convert_examples_to_features_sent_ques(fe, [False, True], 50, 200, tokenizer, is_train=True)
        # dt.create_examples('../../../data/cips-sougou/nottruth/' + i, 'test', False,True,sent=False)
