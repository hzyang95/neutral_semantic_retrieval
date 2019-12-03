import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
from pytorch_transformers import BertTokenizer,BertModel
import torch.nn.utils.rnn as rnn_utils
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# input_ids = []
# for i in ["Hello, my dog is cute","hello"]:
#     input_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]']+tokenizer.tokenize(i)))) # Batch size 1
# input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True)
# outputs = model(input_ids)
# print(outputs)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

LABEL = data.Field(sequential=False, use_vocab=False)

TEXT = data.Field(sequential=False)


train_fields = {'label':('label', LABEL), 'text':('text', TEXT)}

train,val = data.TabularDataset.splits(
        path='../../data/', train='para.dev.json',validation='para.dev.json', format='json',skip_header=True,
        fields=train_fields)

test = data.TabularDataset('test.tsv', format='tsv',skip_header=True,
        fields=[('PhraseId',None),('SentenceId',None),('Phrase', TEXT)])

print(train[0])
print(train[0].__dict__.keys())
print(train[0].label, train[0].text)

TEXT.build_vocab(train)#, max_size=30000)

train_iter = data.BucketIterator(train, batch_size=12, sort_key=lambda x: len(x.text),
                                 shuffle=True,device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=12, sort_key=lambda x: len(x.text),
                                 shuffle=True,device=DEVICE)

batch = next(iter(train_iter))
data = batch.text
label = batch.text
print(batch.text.shape)
print(batch.text)