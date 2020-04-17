import argparse
import os
import sys

import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import BertTokenizer, DistilBertTokenizer, AdamW
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification

from config import set_args
from utils.eval_utils import cal_acc, count_label, cal_prf
from utils.model_utils import load_torch_model
from utils.train_utils import get_data,log_prf_single,pickle_to_data
from data_processor import DataProcessor, convert_examples_to_features_test
sys.path.append('../')

args = set_args()

torch.manual_seed(args.seed)



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

n_gpu = len(args.device.split(','))
# tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')
# model = BertModel.from_pretrained('hfl/rbt3')

def test(model, test_features):
    # test_features = dataset['test']
    test_data = get_data(test_features[:])
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size)
    _ri = []
    _pr = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        _ri.extend(label_ids)
        model.eval()
        if args.model[:3] == 'dis':
            logits = model(input_ids=input_ids, attention_mask=input_mask)[0]
        else:
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        max_prob_batch, pred_batch = torch.max(logits, dim=1)
        _pr.extend(pred_batch)
    ri = 0
    assert len(_pr) == len(_ri)
    _pr = list(_pr)
    _ri = list(_ri)
    # for i in range(len(_pr)):
    #     if _pr[i]==_ri[i]:
    #         ri+=1
    # print(_pr)
    # print(_ri)
    return log_prf_single(_pr, _ri)


def evaluate(top):
    processor = DataProcessor()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.model)

    eval_examples = processor.get_dev_examples('../data/version3_doc_full_dev_all.json', args.test_batch_size)
    eval_features = convert_examples_to_features_test(eval_examples, label_list, args.ans_len, tokenizer,
                                                      verbose=False)
    predictions = []
    targets = []
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for feature in eval_features:
        input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        # Run prediction for full data
        sampler = SequentialSampler(data)
        b_l = input_ids.size(0)
        dataloader = DataLoader(data, sampler=sampler, batch_size=b_l)
        for input_ids, input_mask, segment_ids, label_ids in dataloader:  # , desc="Evaluation")
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()

            with torch.no_grad():
                loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                         labels=label_ids)[:2]
            sigmoid = torch.nn.Sigmoid()
            logits = sigmoid(logits)
            eval_loss += loss.mean().item()
            tops = torch.topk(logits, min(top, b_l), dim=0)
            ind = [0] * b_l
            for ii in tops[1]:
                ind[ii[0]] = 1
            # ind = torch.tensor(ind).to(device)
            #
            # logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').tolist()
            predictions += ind
            targets += label_ids

            nb_eval_examples += b_l
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    return log_prf_single(predictions, targets)



def train(model, dataset):
    train_features = dataset['training']
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)

    train_data = get_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    _max = 0
    best = None
    ind = ''
    all_step = 0

    es = 0
    es_ind = ''
    for i in range(args.epochs):
        print('=======epoch ' + str(i) + '=========')
        for step, batch in enumerate(train_dataloader):
            if args.cuda:
                batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            label_ids = label_ids.unsqueeze(-1)
            model.train()
            if args.model[:3] == 'dis':
                loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
            else:
                loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                     labels=label_ids)[:2]
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (all_step + 1) % args.log_step == 0:
                print('all_step ' + str(all_step) +' epoch ' + str(i) +' step ' + str(step) + '/' + str(len(train_dataloader))
                      + ' loss:' + str(loss))

            if (all_step + 1) % args.eval_step == 0:
                res = test(model, dataset['validation'])
                if res > _max:
                    _max = res
                    best = model
                    ind = str(i) + '_' + str(all_step)
                    es = 0
                else:
                    if i > 1:
                        es += 1
                        if es >= 15:
                            break
                print('all_step '+str(all_step))
                print('ind:' + ind + ' _max:'+str(_max))
                print('early_stop:'+str(es))
            if (all_step + 1) % (args.eval_step*10) == 0:
                save_path = 'best/' + args.task + '_' + str(args.batch_size) + '_' + args.model.replace('/',
                                                    '_') + '_' + str(ind) + '_' + str(_max)[2:6]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    torch.save(best, save_path + '/model.pt')
            all_step += 1
        res = test(model, dataset['validation'])
        if res > _max:
            _max = res
            best = model
            ind = str(i) + '_' + str(all_step)
        print('epoch '+str(i))
        print('ind:' + ind + ' _max:' + str(_max))
        if (i+1) % 3 == 0:
            save_path = 'best/' + args.task + '_' + str(args.batch_size) + '_' + args.model.replace('/',
                          '_') + '_' + str(ind) + '_' + str(_max)[2:6]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                torch.save(best, save_path + '/model.pt')
        if es >= 15:
            print('stop at ' + str(i)+'_'+str(all_step))
            break
    return best, ind
    # print(test(model,dataset))


if __name__ == "__main__":
    # 1. define location to save the model and mkdir if not exists
    # pt = "saved_model/stance_" + args.input
    # if not os.path.exists(pt):
    #     os.mkdir(pt)

    # 2. load dataset
    out_dir = args.out_dir + "/" + args.task + "/"
    dataset = pickle_to_data(out_dir + "/features.pkl")
    print(dataset.keys())
    # dict_keys(['training', 'validation', 'test'])

    # 3. test, proceed, train

    if args.is_test:
        model = load_torch_model(args.save, use_cuda=args.cuda)
        # print(type(model))
        # print(test(model, dataset['test']))
        print(evaluate(3))
    else:
        ''' make sure the folder to save models exist '''
        # if not os.path.exists(args.saved):
        #     os.mkdir(args.saved)

        ''' continue training or not '''
        if args.proceed:
            model = load_torch_model(args.saved, use_cuda=args.cuda)
        else:
            if args.model[:3] == 'dis':
                model = DistilBertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
            else:
                # print(os.path.exists('C:/Users/Yang/Desktop/hfl_rbt3/'))

                model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed_all(args.seed)
                model.cuda()
                if n_gpu > 1:
                    print('let\'s use ' + str(n_gpu) +' GPUs!')
                    model = torch.nn.DataParallel(model)

        best, ind = train(model, dataset)
        testres = test(best, dataset['test'])
        print(testres)
        save_path = 'best/' + args.task + '_final_' + str(args.batch_size) + '_' + args.model.replace('/', '_') + '_' + ind +'_' +str(testres)[2:6]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            torch.save(best, save_path + '/model.pt')
