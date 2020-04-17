import json

_stage = ['train', 'dev', 'test']
stage = _stage[1]

_obj = ['doc', 'para']
obj = _obj[0]

first_num = 3000

# file_path = '../../../preprocessed/' + stage + 'set/search.' + stage + '.json'
file_path = '../../data/sent_train.v1.35000.1205950.0.1.41.json'
file = []
with open(file_path, 'r', encoding='utf-8') as f:
    file += f.readlines()
# with open('trainset/zhidao.train.json','r',encoding = 'utf-8') as f:
#     file += f.readlines()


print('len(file)'+str(len(file[:])))
label = {0:0,1:0}
for line in file[:1000000]:
    js = json.loads(line)
    # ks = list(js.keys())
    # print(ks)
    # print(js['label'])
    label[js['label']]+=1

print(label)
# dc_num = 0
# dc_len = 0
# dc_len_l = []
# dc_a_num = 0
#
# para_num = 0
# para_len = 0
# para_len_l = []
# para_a_num = 0
#
# le = []
# sc = []
# mx = 0
# mxa = 0
# for line in file:
#     js = json.loads(line)
#     ks = list(js.keys())
#     # print(ks)
#     dcms = js['documents']
#     le.append(len(dcms))
#     mxa = max(mxa, len(dcms))
#     res = []
#     for dc in dcms:
#         dc_a_num += 1
#         paras = dc['paragraphs']
#         gold = dc["most_related_para"]
#         d_l = 0
#         for i, para in enumerate(paras):
#             p_l = len(para)
#             d_l += p_l
#             para_a_num += 1
#             if i == gold:
#                 # if p_l<15:print(para)
#                 para_len_l.append(p_l)
#                 para_len += p_l
#                 para_num += 1
#         if dc["is_selected"]:
#             dc_len_l.append(d_l)
#             dc_len += d_l
#             dc_num += 1
#             res.append(1)
#     sc.append(len(res))
#
#     mx = max(mx, len(res))
#     # print(js['question_type'])
#
# # para_len_l.sort()
# # dc_len_l.sort()
# print(dc_a_num)
# print(dc_num)
# print(dc_len / dc_num)
# # print(dc_len_l)
# print(para_a_num)
# print(para_num)
# print(para_len / para_num)
# # print(para_len_l)
# print(len(sc))
# print(sc)
# print(len(le))
# print(le)
# for i in range(mx+1):
#     print(str(i) + ' ' + str(sc.count(i)))
# for i in range(mxa+1):
#     print(str(i) + ' ' + str(le.count(i)))
