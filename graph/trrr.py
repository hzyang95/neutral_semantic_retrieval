import argparse
import os
import sys
import time

import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import BertTokenizer, DistilBertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification

from graph_model import GraphBasedModel, DisGraphBasedModel
from config import set_args
from utils.train_utils import log_prf_single, pickle_to_data, data_to_pickle, process_logit

from data_processor import DataProcessor, convert_examples_to_features

from graph_utils import load_torch_model, get_batches

import logging

from data_processor import convert_examples_to_features_sent
from graph_utils import get_batches_sent

t = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-6s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='mt_' + t + '.log',  # ' + t + '
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
# 设置格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
# 告诉handler使用这个格式
console.setFormatter(formatter)
# add the handler to the root logger
# 为root logger添加handler
logging.getLogger('').addHandler(console)

logging.info('begin')

sys.path.append('../')

args = set_args()
logging.info(args)

torch.manual_seed(args.seed)
logging.info('seed: ' + str(args.seed))

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

n_gpu = len(args.device.split(','))

logging.info('n_gpu:' + str(n_gpu))


# tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')
# model = BertModel.from_pretrained('hfl/rbt3')

def test(model, tokenizer, test_data):
    # model = model.module
    _ri = []
    _pr = []
    # return
    all_step = 0
    loss = 0.0
    logging.info(args.test_batch_size)
    test_data = DataLoader(test_data, batch_size=args.test_batch_size)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data)):
            if step < 5:
                print(len(batch))
            if args.cuda:
                batch = tuple(t.cuda() for t in batch)

            if args.sent:
                input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label, sent_num = batch

                inputs = {'input_ids': input_ids,
                          'input_mask': input_mask,
                          'segment_ids': segment_ids,  # XLM don't use segment_ids
                          'adj_matrix': adj_matrix,
                          'graph_mask': input_graph_mask,
                          'sent_start': sent_start,
                          'sent_end': sent_end,
                          'sent_sum_way': args.sent_sum_way,
                          'sp_label': input_sp_label,
                          'sent_num': sent_num,
                          'gtem': 'sent'
                          }
            else:
                input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label = batch

                inputs = {'input_ids': input_ids,
                          'input_mask': input_mask,
                          'segment_ids': segment_ids,  # XLM don't use segment_ids
                          'adj_matrix': adj_matrix,
                          'graph_mask': input_graph_mask,
                          'sent_start': sent_start,
                          'sent_end': sent_end,
                          'sent_sum_way': args.sent_sum_way,
                          'sp_label': input_sp_label,
                          }

            # print(input_sp_label)
            # print(sent_start)

            _sent_len = []
            for idx, data in enumerate(input_sp_label):
                # ind = [0] * len(sent_start[idx])
                # for ii in data.detach().cpu().numpy().tolist():
                #     ind[int(ii)] = 1
                sent_len = 0
                if args.sent:
                    sent_len = int(sent_num[idx])
                else:
                    for _i in sent_start[idx].tolist():
                        if _i != -1:
                            sent_len += 1
                        else:
                            break
                _sent_len.append(sent_len)
                ind = data.detach().cpu().numpy().tolist()[:sent_len]
                ind = [int(i) for i in ind]
                # print(ind)
                _ri.extend(ind)
            example_indices = torch.arange(all_step, all_step + batch[0].size(0))
            all_step += batch[0].size(0)
            model.eval()
            outputs = model(**inputs)
            # if args.model[:3] == 'dis':
            #     logits = model(input_ids=input_ids, attention_mask=input_mask)[0]
            # else:
            #     logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
            # max_prob_batch, pred_batch = torch.max(logits, dim=1)

            # pred_batch = torch.sigmoid(outputs)
            pred_batch = process_logit(example_indices, outputs[1], _sent_len)
            _loss = outputs[0]
            if n_gpu > 1:
                _loss = _loss.mean()  # mean() to average on multi-gpu.
            loss += _loss.item()
            _pr.extend(pred_batch)
    loss = float(loss) / float(all_step)
    ri = 0
    # print(_pr)
    # print(_ri)
    assert len(_pr) == len(_ri)
    _pr = list(_pr)
    _ri = list(_ri)
    # for i in range(len(_pr)):
    #     if _pr[i]==_ri[i]:
    #         ri+=1
    # print(_pr)
    # print(_ri)
    return loss, log_prf_single(_pr, _ri)


def train(model, tokenizer, dataset):
    train_batches = dataset['train']

    train_sampler = RandomSampler(train_batches)
    train_dataloader = DataLoader(train_batches, sampler=train_sampler, batch_size=args.batch_size)

    _max = 0
    _min = 9999999999
    best = None
    ind = ''
    all_step = 0

    es = 0
    _t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_steps * _t_total), _t_total)

    t_total = len(train_dataloader)* args.epochs
    aver_loss = 0.0
    for i in range(args.epochs):
        logging.info('=======epoch ' + str(i) + '=========')
        for step, batch in enumerate(train_dataloader):
            all_step += 1
            if args.cuda:
                batch = tuple(t.cuda() for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            # label_ids = label_ids.unsqueeze(-1)
            model.train()
            if args.sent:
                input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label, sent_num = batch

                inputs = {'input_ids': input_ids,
                          'input_mask': input_mask,
                          'segment_ids': segment_ids,  # XLM don't use segment_ids
                          'adj_matrix': adj_matrix,
                          'graph_mask': input_graph_mask,
                          'sent_start': sent_start,
                          'sent_end': sent_end,
                          'sent_sum_way': args.sent_sum_way,
                          'sp_label': input_sp_label,
                          'sent_num': sent_num,
                          'gtem': 'sent'
                          }
            else:
                input_ids, input_mask, segment_ids, adj_matrix, input_graph_mask, sent_start, sent_end, input_sp_label = batch

                inputs = {'input_ids': input_ids,
                          'input_mask': input_mask,
                          'segment_ids': segment_ids,  # XLM don't use segment_ids
                          'adj_matrix': adj_matrix,
                          'graph_mask': input_graph_mask,
                          'sent_start': sent_start,
                          'sent_end': sent_end,
                          'sent_sum_way': args.sent_sum_way,
                          'sp_label': input_sp_label,
                          }

            outputs = model(**inputs)
            loss = outputs[0]

            # if args.model[:3] == 'dis':
            #     loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
            # else:
            #     loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            #                          labels=label_ids)[:2]
            # print(loss)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            try:
                loss.backward()
            except:
                print(input_ids)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            aver_loss += float(loss)
            # print(loss)
            if (all_step) % args.log_step == 0:
                logging.info('all_step ' + str(all_step) + '/' + str(t_total) + ' epoch ' + str(i) + ' step ' + str(
                    step) + '/' + str(
                    len(train_dataloader))
                             + ' loss:' + str(float(aver_loss) / float(all_step)))

            if (all_step) % args.eval_step == 0:
                eval_loss, res = test(model, tokenizer, dataset['dev'])
                # if res > _max:
                #     _max = res
                #     best = model
                #     ind = str(i) + '_' + str(all_step)
                if eval_loss < _min:
                    _min = eval_loss
                    _max = res
                    best = model
                    ind = str(i) + '_' + str(all_step)
                    es = 0
                else:
                    if i > 1:
                        es += 1
                        if es >= 15:
                            break
                logging.info('now_best epoch_allstep: ' + ind + ' maxf1:' + str(_max))
                logging.info('now_best epoch_allstep: ' + ind + ' minloss:' + str(_min))
                logging.info('early_stop:' + str(es))
            if (all_step) % (args.eval_step * 10) == 0:
                save_path = 'best/' + args.task + '_' + str(args.batch_size) + '_' + args.model.replace('/',
                                                                                                        '_') + '_' + str(
                    ind) + '_' + str(_min)[2:6] + '_' + str(_max)[2:6]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    torch.save(best, save_path + '/model.pt')

        eval_loss, res = test(model, tokenizer, dataset['dev'])
        # if res > _max:
        #     _max = res
        #     best = model
        #     ind = str(i) + '_' + str(all_step)
        if eval_loss < _min:
            _min = eval_loss
            _max = res
            best = model
            ind = str(i) + '_' + str(all_step)

        logging.info('epoch ' + str(i))
        logging.info('now_best epoch_allstep: ' + ind + ' maxf1:' + str(_max))
        logging.info('now_best epoch_allstep: ' + ind + ' minloss:' + str(_min))
        if (i + 1) % 3 == 0:
            save_path = 'best/' + args.task + '_' + str(args.batch_size) + '_' + args.model.replace('/',
                                                                                                    '_') + '_' + str(
                ind) + '_' + str(_min)[2:6] + '_' + str(_max)[2:6]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                torch.save(best, save_path + '/model.pt')
        if es >= 15:
            logging.info('stop at ' + str(i) + '_' + str(all_step))
            break
    return best, ind
    # print(test(model,dataset))


if __name__ == "__main__":
    # 1. define location to save the model and mkdir if not exists
    # pt = "saved_model/stance_" + args.input
    # if not os.path.exists(pt):
    #     os.mkdir(pt)
    # tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')
    if args.sent:
        args.max_seq_length = 150
    if args.model[:3] == 'dis':
        tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)

    # 2. load dataset
    if not args.is_test:
        if args.raw:
            # dict_keys(['training', 'validation', 'test'])
            processor = DataProcessor()
            T = False
            dataset = {'train': processor.create_examples(args.train_data, 'train', T, sent=args.sent),
                       'dev': processor.create_examples(args.dev_data, 'dev', T, sent=args.sent),
                       'test': processor.create_examples(args.test_data, 'test', T, sent=args.sent)}
            # 3. test, proceed, train
            for i in dataset:
                logging.info(i + ' length: ' + str(len(dataset[i])))
                if args.sent:
                    features = convert_examples_to_features_sent(
                        dataset[i], [False, True], args.max_ques_length, args.max_seq_length, tokenizer, is_train=True)
                    data = get_batches_sent(features, is_train=True)
                else:
                    features = convert_examples_to_features(
                        dataset[i], [False, True], args.max_ques_length, args.max_seq_length, tokenizer, is_train=True)
                    data = get_batches(features, is_train=True)
                dataset[i] = data
            # data_to_pickle(dataset, args.out_dir + "/test_features_full" + args.model[:3] +".pkl")
        else:
            dataset = pickle_to_data("data/test_features_full" + args.model[:3] + ".pkl")

    if args.is_test:
        print(args.sent)
        processor = DataProcessor()
        dt = processor.create_examples(args.test_data, 'test', False, sent=args.sent)
        # features = convert_examples_to_features(
        #     dt, [False, True], args.max_ques_length, args.max_seq_length, tokenizer, is_train=True)
        # data = get_batches(features, is_train=True)
        # dataset_ = DataLoader(data, batch_size=args.test_batch_size)
        # data_to_pickle(dataset_, args.out_dir + "/features_test_3000.pkl")
        if args.sent:
            features = convert_examples_to_features_sent(
                dt, [False, True], args.max_ques_length, args.max_seq_length, tokenizer, is_train=True)
            data = get_batches_sent(features, is_train=True)
        else:
            features = convert_examples_to_features(
                dt, [False, True], args.max_ques_length, args.max_seq_length, tokenizer, is_train=True)
            data = get_batches(features, is_train=True)
        model = load_torch_model(args.save, use_cuda=args.cuda)
        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed_all(args.seed)
                model.cuda()
                if n_gpu > 1:
                    print('let\'s use ' + str(n_gpu) + ' GPUs!')
                    model = torch.nn.DataParallel(model)
        # print(type(model))
        # model = None
        # ['test']
        print(test(model, tokenizer, data))
    else:
        ''' make sure the folder to save models exist '''
        # if not os.path.exists(args.saved):
        #     os.mkdir(args.saved)

        ''' continue training or not '''
        if args.proceed:
            model = load_torch_model(args.save, use_cuda=args.cuda)
        else:
            # model = GraphBasedModel.from_pretrained(args.model, num_rel=1, pretrain=args.model)
            if args.model[:3] == 'dis':
                # tokenizer = DistilBertTokenizer.from_pretrained(args.model)
                model = DisGraphBasedModel.from_pretrained(args.model, num_rel=1)
            else:
                model = GraphBasedModel.from_pretrained(args.model, num_rel=1)
                # print(os.path.exists('C:/Users/Yang/Desktop/hfl_rbt3/'))
                # tokenizer = BertTokenizer.from_pretrained(args.model)
        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed_all(args.seed)
                model.cuda()
                if n_gpu > 1:
                    print('let\'s use ' + str(n_gpu) + ' GPUs!')
                    model = torch.nn.DataParallel(model)

        best, ind = train(model, tokenizer, dataset)
        min_loss, testres = test(best, tokenizer, dataset['test'])
        print(min_loss)
        print(testres)
        save_path = 'best/' + args.task + '_final_' + str(args.batch_size) + '_' + args.model.replace('/',
                                                                                                      '_') + '_' + ind + '_' + str(
            min_loss)[2:6] + '_' + str(testres)[2:6]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            torch.save(best, save_path + '/model.pt')
