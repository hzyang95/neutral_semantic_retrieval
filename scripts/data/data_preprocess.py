import json

_stage = ['train', 'dev', 'test']
stage = _stage[1]

_obj = ['doc', 'para']
obj = _obj[1]

first_num = 20000

file_path = '../../data/preprocessed/' + stage + 'set/search.' + stage + '.json'

save_path = '../../data/' + obj + '.' + stage + '.json'

file = []
with open(file_path, 'r', encoding='utf-8') as f:
    file += f.readlines()

data = []
for line in file[:first_num]:
    js = json.loads(line)
    dcms = js['documents']
    ques = js['question']
    for dc in dcms:
        paras = dc['paragraphs']
        gold = dc["most_related_para"]
        if obj == 'para':
            for i, para in enumerate(paras):
                text = para
                label = 0
                if i == gold:
                    label = 1
                data.append({"label": label, 'question':ques,'text': text})
        if obj == 'doc':
            text = ' '.join(paras)
            label = 0
            if dc["is_selected"]:
                label = 1
            data.append({"label": label, 'question':ques, 'text': text})

with open(save_path, 'w', encoding='utf-8') as json_file:
    for each_dict in data:
        json_file.write(json.dumps(each_dict) + '\n')
