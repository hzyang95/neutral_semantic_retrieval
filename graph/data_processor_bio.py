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


class DataProcessorBio(object):

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
            gold = row['gold']
            content = []
            label = []
            ll = 0
            # guid+=1
            # if row['top'] == 0:
            #     lb0 += 1
            for item in row['sample']:
                lll = len(item['text'])
                if lll < 5:
                    neg.append(question + ':    ' + item['text'])
                    continue
                if lll > 200:
                    # if lll < 300:
                    neg_long.append(question + ':   ' + item['text'])
                    continue
                if (not sent and (ll + lll > 400 or len(content) + 1 >= 30)) or(sent and(len(content) + 1 >= 50)):
                    if len(content) == 0:
                        continue
                    len_of_cont = len(content)
                    _aver[len_of_cont // 10] += 1
                    all_num +=len_of_cont
                    examples.append(InputExample(guid=guid, question=question, answer=content, label=gold))
                    guid+=1
                    # print(guid)
                    
                    content = []
                    ll = 0

                ll += lll
                content.append(item['text'])
            # print(len(content))
            if len(content) == 0:
                continue
            len_of_cont = len(content)
            _aver[len_of_cont // 10] += 1
            all_num += len_of_cont
            examples.append(InputExample(guid=guid, question=question, answer=content, label=gold))
            guid+=1
            # print(guid)
        print(_aver)
        print(all_num)
        print(len(examples))
        logging.info(str(set_type) + str(len(examples)))
        return examples


if __name__ == "__main__":
    dataset = [
        '/users8/hzyang/proj/neutral_semantic_retrieval/scripts/bioasq/bioasq_test.552.0.0.106.73.json'
    ]
    dt = DataProcessorBio()
    for i in dataset:
        # dt.create_examples('../data/' + i, 'test', False)
        # dt.create_examples('../../../data/cips-sougou/nottruth/' + i, 'test', False,True,sent=False)
        dt.create_examples(i, 'test', False,True,sent=True)
