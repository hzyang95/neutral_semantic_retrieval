import json

from tqdm import tqdm

from ner_and_np import get_NER, get_NER_LTP, get_NP


def convert(path, test=True):
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
    if test:
        file = file[:2]
    print(len(file))
    new_data = []
    _aver = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, line in enumerate(tqdm(file[:])):
        _row = json.loads(line)
        ans_data = []
        question = _row['question']
        q_ner = []
        q_ner_ltp = []
        q_np = []
        q_ner = get_NER(question)
        q_ner_ltp = get_NER_LTP(question)
        q_np = get_NP(question)
        for doc_id, row in enumerate(_row['examples']):
            # print(row)
            temp_data = row.copy()
            temp_data['sample'] = []
            for item in row['sample']:
                lll = item.copy()
                # print(lll)
                sent = item['text']

                a_ner = []
                a_ner_ltp = []
                a_np = []
                # if i == 11916:
                #     # print(sent)
                a_ner = get_NER(sent)
                a_ner_ltp = get_NER_LTP(sent)
                a_np = get_NP(sent)

                lll['a_ner'] = a_ner
                lll['a_ner_ltp'] = a_ner_ltp
                lll['a_np'] = a_np
                # print(lll)
                temp_data['sample'].append(lll)
            # print(temp_data)
            ans_data.append(temp_data)
        new_items = {'question': question, 'q_ner': q_ner, 'q_ner_ltp': q_ner_ltp,
                     'q_np': q_np, 'examples': ans_data}
        # print(new_items)
        new_data.append(new_items)
    with open(path + '.ety', 'w', encoding='utf-8') as json_file:
        for each_dict in new_data:
            json_file.write(json.dumps(each_dict) + '\n')


if __name__ == "__main__":
    dataset = [
        # 'data_for_graph_train.v1.30000.143127.2.8.41.json',
        # 'data_for_graph_valid.2000.10115.3.7.41.json',
        # 'data_for_graph_test.3000.13848.3.9.40.json',
        'ques_data_for_graph_train.v1.30000.27127.2.8.41.json',
        'ques_data_for_graph_valid.2000.10115.1809.3.7.41.json',
        'ques_data_for_graph_test.3000.13848.2621.3.9.40.json'
    ]
    for i in dataset:
        convert('../../data/' + i, False)
