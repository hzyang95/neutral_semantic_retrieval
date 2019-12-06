import json

_stage = ['train', 'dev', 'test']
stage = _stage[1]

_obj = ['doc', 'para']
obj = _obj[1]


_source =['search','zhidao']
source = _source[0]


first_num = 1000

file_path = '../../../preprocessed/' + stage + 'set/'+ source +'.' + stage + '.json'

save_path = '../../data/version2.' + obj + '.' + source +'.' + stage + '.json'

file = []
with open(file_path, 'r', encoding='utf-8') as f:
    file += f.readlines()

data = []
tt=0
for line in file[:first_num]:
    js = json.loads(line)
    dcms = js['documents']
    ques = js['question']
    if obj == 'doc':
        temp_data = {'question':ques,'sample':[],'top':0}
    cand = 0
    for dc in dcms:
        paras = dc['paragraphs']
        gold = dc["most_related_para"]
        if obj == 'para':
            temp_data = {'question': ques, 'sample': [], 'top': 0}
            for i, para in enumerate(paras):
                tt += 1
                text = para
                label = 0
                if i == gold:
                    label = 1
                    cand += 1
                temp_data['sample'].append({"label": label, 'question':ques,'text': text})
            temp_data['top'] = cand
            data.append(temp_data)

        if obj == 'doc':
            text = ' '.join(paras)
            label = 0
            if dc["is_selected"]:
                label = 1
                cand += 1
            temp_data['sample'].append({"label": label, 'question':ques, 'text': text})
    if obj == 'doc':
        temp_data['top'] = cand
        data.append(temp_data)
print(len(data))
print(tt)
with open(save_path, 'w', encoding='utf-8') as json_file:
    for each_dict in data:
        json_file.write(json.dumps(each_dict) + '\n')
