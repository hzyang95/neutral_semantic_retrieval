f = open('1_para_200000_1000_100_50_1_gpupara_new', 'r', encoding='utf-8')
# f = open('1_tops4_sent_1000000_5000_500_250_1_gpu2para_new', 'r', encoding='utf-8')


arr = []

l1=[]
l2=[]
l3=[]

for i in f.read().split('\n')[45:]:
    arr.append(i[60:])



    if i[60:60+16]=='  global_step = ':
        # print(i[60:60 + 16])
        # print(i[60 + 16:].strip())
        l1.append(int(i[60+16:].strip()))
    if i[60:60+15]=='  train_loss = ':
        # print(i[60:60 + 15])
        # print(i[60+15:].strip())
        l2.append(float(i[60+15:].strip()))
    if i[60:60+14]=='  eval_loss = ':
        print(i[60:60 + 14])
        print(i[60 + 14:].strip())
        l3.append(float(i[60+14:].strip()))

print(arr)
print(len(l1))
print(len(l2))
print(len(l3))
print(l1)
print(l2)
print(l3)
print(len('2019-12-24 16:27:50,722 - train.py - <module>- 326- -INFO -'))
print(len('  global_step = '))
print(len('  train_loss = '))
print(len('  eval_loss = '))

from torch.utils.tensorboard import SummaryWriter

tb_writer1 = SummaryWriter(log_dir='../tb/para/train')
# tb_writer2 = SummaryWriter(log_dir='../tb/sent/eval')


for i,lo in enumerate(l1):
    tb_writer1.add_scalar('loss',  l2[i] , lo)
    # tb_writer2.add_scalar('loss', l3[i], lo)