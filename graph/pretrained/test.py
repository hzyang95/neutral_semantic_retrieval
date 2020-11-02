# import heapq

# h = []
# ar = [(3, (1, 3)), (1, 2)]
# for i in ar:
#     heapq.heappush(h, i)
# print(h)

# di = {'1': 1, '3': 2, '2': 3}

# print(sorted(di.items(), key=lambda x: x[0]))
# print(sorted(di.items(), key=lambda x: x[1]))

from transformers import RobertaModel
model = RobertaModel.from_pretrained('roberta-base')