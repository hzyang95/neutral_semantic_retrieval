from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import random
import json
from tqdm import tqdm, trange
from config import set_args

from data_processor import DataProcessor, convert_examples_to_features, convert_examples_to_features_test
# from tensorboardX import SummaryWriter

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, DistilBertTokenizer
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification

from pytorch_pretrained_bert.optimization import BertAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"

# 0 ： mul
# 1 : dis
# 2 : chi

dis = 2
tops = 1
args = set_args()
if dis == 1:
    # args.train_batch_size = 100
    # args.eval_batch_size = 50
    args.bert_model = 'distilbert-base-multilingual-cased'
if dis == 2:
    # args.bert_model = 'bert-base-chinese'
    args.bert_model = 'hfl/rbt3'

if args.gra == 'sent':
    # args.train_data = '../data/sent_train.v1.35000.1205950.0.1.41.json'
    # args.dev_data = '../data/sent_valid.10000.23421.3.8.41.json'
    args.train_data = '../data/cmrc2018__train.10142.10.21.41.json'
    args.dev_data = '../data/cmrc2018__dev.1548.1.12.40.json'
    args.max_seq_length = 150
    args.train_batch_size *= 5
    args.eval_batch_size *= 5
    # args.train_num *= 5
    # args.dev_num *= 5
    args.train_num = 10142
    args.dev_num = 1548
    tops = 2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
formatter_file = logging.Formatter('%(asctime)s - %(filename)s '
                                   '- %(funcName)s- %(lineno)d- '
                                   '-%(levelname)s - %(message)s')
logger = logging.getLogger()
file = logging.FileHandler(str(dis) + '_up_' + str(args.gra) + '_' + str(args.train_num) + '_' + str(args.dev_num) +
                           '_' + str(args.train_batch_size) + '_' + str(args.eval_batch_size) + '_' +
                           str(torch.cuda.device_count()) + '_gpu'+os.environ["CUDA_VISIBLE_DEVICES"]+'para_new', encoding='utf-8')
file.setLevel(level=logging.INFO)
file.setFormatter(formatter_file)
logger.addHandler(file)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def add_figure(name, writer, global_step, train_loss, test_loss, pre, rec, f1):  # , test_acc
    writer.add_scalars(name + '_data/loss_group',
                       {'train_loss': train_loss, 'test_loss': test_loss, 'pre': pre, 'rec': rec, 'f1': f1},
                       global_step)
    # writer.add_scalar(name + '_data/test_acc', test_acc, global_step)
    return


# def evaluate(do_pred=False, pred_path=None):
#     logger.info("***** Running evaluation *****")
#     logger.info("  Num examples = %d", len(eval_examples))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
#     eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
#     eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
#     eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
#     eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
#     # Run prediction for full data
#     eval_sampler = SequentialSampler(eval_data)
#     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#
#     model.eval()
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     predictions = []
#     for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluation"):
#         input_ids = input_ids.cuda()
#         input_mask = input_mask.cuda()
#         segment_ids = segment_ids.cuda()
#         label_ids = label_ids.cuda().unsqueeze(-1)
#
#         with torch.no_grad():
#             if dis:
#                 tmp_eval_loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
#             else:
#                 tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids,
#                                               attention_mask=input_mask, labels=label_ids)[:2]
#             # logits = model(input_ids, segment_ids, input_mask)
#
#         logits = logits.detach().cpu().numpy()
#         label_ids = label_ids.to('cpu').numpy()
#         # tmp_eval_accuracy = accuracy(logits, label_ids)
#
#         predictions.append(logits)
#
#         eval_loss += tmp_eval_loss.mean().item()
#         # eval_accuracy += tmp_eval_accuracy
#
#         nb_eval_examples += input_ids.size(0)
#         nb_eval_steps += 1
#
#     eval_loss = eval_loss / nb_eval_steps
#     # eval_accuracy = eval_accuracy / nb_eval_examples
#
#     result = {'eval_loss': eval_loss,
#               # 'eval_accuracy': eval_accuracy,
#               'global_step': global_step}
#
#     logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
#         # writer.write("%s = %s\n" % (key, str(result[key])))
#
#     # if do_pred and pred_path is not None:
#     #     logger.info("***** Writting Predictions ******")
#     #     logits0 = np.concatenate(predictions, axis=0)[:, 0]
#     #     ground_truth = [fea.label_id for fea in eval_features]
#     #     pandas.DataFrame({'logits0': logits0, 'label': ground_truth}).to_csv(pred_path)
#     return eval_loss  # , eval_accuracy
def calc_result(ref, res):
    assert len(ref) == len(res)
    length = len(ref)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(length):
        a = ref[i]
        b = res[i]
        if a == 1:
            if b == 1:
                tp += 1
            else:
                fn += 1
        else:
            if b == 1:
                fp += 1
            else:
                tn += 1
    # print(path)
    # print("tp: " + str(tp))
    # print("fp: " + str(fp))
    # print("tn: " + str(tn))
    # print("fn: " + str(fn))
    # print("acc: " + str((tp + tn) / (tp + fp + tn + fn)))
    pr = tp / (tp + fp)
    # print("pre: " + str(pr))
    rec = tp / (tp + fn)
    # print("rec: " + str(rec))
    f1 = 2 * pr * rec / (pr + rec)
    # print("f1:" + str(f1))
    return pr, rec, f1


def evaluate(top, dis):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    predictions = []
    targets = []
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for feature in eval_features:
        input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.float)
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
                if dis == 1:
                    loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
                else:
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
    pre, rec, f1 = calc_result(targets, predictions)

    pre = round(pre, 4)
    rec = round(rec, 4)
    f1 = round(f1, 4)
    eval_loss = round(eval_loss, 4)

    result = {'eval_loss': eval_loss,
              'pre': pre,
              'rec': rec,
              'f1': f1}
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return eval_loss, pre, rec, f1


if __name__ == "__main__":

    if dis == 1:
        eval_step = 250
    else:
        eval_step = 50

    # writer = SummaryWriter(log_dir="figures")

    # Set GPU Issue
    n_gpu = torch.cuda.device_count()
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    logger.info("n_gpu: {} Grad_Accum_steps: {} Batch_size: {}".format(
        n_gpu, args.gradient_accumulation_steps, args.train_batch_size))

    # Set Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    num_labels = 1
    processor = DataProcessor()
    label_list = processor.get_labels()

    # Prepare Tokenizer
    if dis == 1:
        tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare Model
    if dis == 1:
        model = DistilBertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)

    model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare Optimizer
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data, args.train_num)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    # Training
    eval_examples = processor.get_dev_examples(args.dev_data, args.dev_num)
    eval_features = convert_examples_to_features_test(eval_examples, label_list, args.max_seq_length, tokenizer,
                                                      verbose=False)

    global_step = 0
    m_loss = 1000000000
    max_f1 = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, verbose=True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epc in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("-----------***** training epoch {}*****----------".format(epc))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                label_ids = label_ids.unsqueeze(-1)
                # print(input_ids.size())
                # print(input_mask.size())
                # print(segment_ids.size())
                # print(label_ids.size())
                if dis == 1:
                    loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
                else:
                    loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                         labels=label_ids)[:2]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    # Save a trained model
                    if global_step % eval_step == 0 and global_step != 0:
                        result = {'train_loss': tr_loss / nb_tr_steps,
                                  # 'eval_accuracy': eval_accuracy,
                                  'global_step': global_step}

                        logger.info("***** train results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                        output_prediction_file = os.path.join(args.output_dir, "smallbs_up_{}_{}_{}_{}_{}_top{}_epoch{}_step{}_pred.csv".
                                                              format(int(dis), args.gra, args.train_num, args.dev_num,
                                                                     args.name, str(tops), epc, global_step))
                        # eval_loss = evaluate(do_pred=True, pred_path=output_prediction_file)
                        eval_loss, pre, rec, f1 = evaluate(tops, dis)
                        model.train()
                        if max_f1 < f1:
                            max_f1 = f1
                            # add_figure(args.name, writer, global_step, tr_loss / nb_tr_steps, eval_loss, pre, rec, f1)
                            output_model_file = os.path.join(args.ckpt_dir,
                                                             "smallbs_up_{}_{}_{}_{}_{}_top{}_epoch{}_step{}_f1{}_loss{}.bin".format(
                                                                 int(dis), args.gra, args.train_num, args.dev_num,
                                                                 args.name, str(tops),
                                                                 epc, global_step, f1, eval_loss))
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            torch.save(model_to_save.state_dict(), output_model_file)
                    # add_figure(args.name, writer, global_step, tr_loss / nb_tr_steps, eval_loss)
                    # if m_loss > eval_loss:
                    #     m_loss = eval_loss
                    #     output_model_file = os.path.join(args.ckpt_dir,
                    #                                      "{}_{}_{}_{}_{}_epoch{}_step{}_ckpt_loss{}.bin".format(
                    #                                          int(dis), args.gra, args.train_num, args.dev_num, args.name,
                    #                                          epc, global_step, eval_loss))
                    #     model_to_save = model.module if hasattr(model,
                    #                                             'module') else model  # Only save the model it-self
                    #     torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    # model.cuda()
