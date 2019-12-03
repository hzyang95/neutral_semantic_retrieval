import json

_stage = ['train', 'dev', 'test']
stage = _stage[1]

_obj = ['doc', 'para']
obj = _obj[0]

first_num = 3000

file_path = '../../../preprocessed/' + stage + 'set/search.' + stage + '.json'

save_path = '../../data/version2.' + obj + '.' + stage + '.json'

file = []
with open(file_path, 'r', encoding='utf-8') as f:
    file += f.readlines()

data = []
tt=0
for line in file[:first_num]:
    js = json.loads(line)
    dcms = js['documents']
    ques = js['question']
    temp_data = {'question':ques,'sample':[],'top':0}
    cand = 0
    for dc in dcms:
        paras = dc['paragraphs']
        gold = dc["most_related_para"]
        if obj == 'para':
            for i, para in enumerate(paras):
                text = para
                label = 0
                if i == gold:
                    label = 1
                    cand += 1
                temp_data['sample'].append({"label": label, 'question':ques,'text': text})
        if obj == 'doc':
            tt+=1
            text = ' '.join(paras)
            label = 0
            if dc["is_selected"]:
                label = 1
                cand += 1
            temp_data['sample'].append({"label": label, 'question':ques, 'text': text})
    temp_data['top'] = cand
    data.append(temp_data)
print(len(data))
print(tt)
with open(save_path, 'w', encoding='utf-8') as json_file:
    for each_dict in data:
        json_file.write(json.dumps(each_dict) + '\n')
