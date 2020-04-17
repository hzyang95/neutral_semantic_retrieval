import json

from tqdm import tqdm

_stage = ['train', 'dev', 'test']
stage = _stage[1]

_obj = ['doc', 'para']
obj = _obj[0]


# _source =['search','zhidao']
# source = _source[1]

ssource = 'full'

first_num = 1000
# first_num = 'all'
otp = 1


file = []


for source in ['search','zhidao']:
    file_path = '../../../preprocessed/' + stage + 'set/'+ source +'.' + stage + '.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        file += f.readlines()[:]

print(len(file))

data = []
for line in tqdm(file[:]):
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

print(len(data))

save_path = '../../data/version3_' + obj + '_' + ssource + '_' + stage + '_' + str(len(data)) + '.json'


with open(save_path, 'w', encoding='utf-8') as json_file:
    for each_dict in data:
        json_file.write(json.dumps(each_dict) + '\n')






