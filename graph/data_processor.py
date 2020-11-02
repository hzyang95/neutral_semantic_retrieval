import json
import logging

from tqdm import tqdm

from utils.train_utils import InputExample

logger = logging.getLogger(__name__)


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


class DataProcessor(object):

    def get_labels(self):
        return [False, True]

    def create_examples(self, path, set_type, test=False, is_train=True, sent=False):
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
        all_num=0
        # lb0 = 0
        guid=0
        for i, line in enumerate(tqdm(file[:])):
            row = json.loads(line)
            question = row['question']
            content = []
            label = []
            ll = 0
            guid+=1
            # if row['top'] == 0:
            #     lb0 += 1
            for item in row['sample']:
                lll = len(item['text'])
                # print(ll)
                # if check_contain_chinese(item['text']) is False:
                #     continue
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
                # answer = '{} {}'.format(row['context'], row['title'])
                if (not sent and (ll + lll > 400 or len(content) + 1 >= 30)) or(sent and(len(content) + 1 >= 50)):
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
                    all_num +=len_of_cont
                    examples.append(InputExample(guid=guid, question=question, answer=content, label=label))
                    guid+=1
                    content = []
                    label = []
                    ll = 0

                ll += lll
                content.append(item['text'])
                label.append(item['label'])
            # print(len(content))
            if len(label) == 0:
                continue
            assert len(content) == len(label)
            len_of_cont = len(content)
            _aver[len_of_cont // 10] += 1
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
            examples.append(InputExample(guid=guid, question=question, answer=content, label=label))
        print(_aver)
        print(all_num)
        # print(len(neg))
        # # print('\n'.join(neg[:10]))
        # print(len(neg_long))
        # # print('\n'.join(neg_long[:10]))
        # # print(nnnn)
        # # print(lb0)
        print(len(examples))
        logging.info(str(set_type) + str(len(examples)))
        return examples


if __name__ == "__main__":
    dataset = [
        'data_for_graph_train.v1.30000.143127.2.8.41.json',
        'data_for_graph_valid.2000.10115.3.7.41.json',
        'data_for_graph_test.3000.13848.3.9.40.json'
    ]
    dt = DataProcessor()
    for i in dataset:
        # dt.create_examples('../data/' + i, 'test', False)
        # dt.create_examples('../../../data/cips-sougou/nottruth/' + i, 'test', False,True,sent=False)
        dt.create_examples('../data/' + i, 'test', False,True,sent=False)
