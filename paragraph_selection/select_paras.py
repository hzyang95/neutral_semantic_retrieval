from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import pandas
from tqdm import tqdm
from config import set_args
from data_processor import DataProcessor,convert_examples_to_features,convert_examples_to_features_test
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import Counter
from transformers import BertTokenizer,DistilBertTokenizer
from transformers import BertForSequenceClassification,DistilBertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(torch.cuda.device_count())


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def result(ref, res):
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
    print(path)
    print("tp: " + str(tp))
    print("fp: " + str(fp))
    print("tn: " + str(tn))
    print("fn: " + str(fn))
    print("acc: " + str((tp + tn) / (tp + fp + tn + fn)))
    pr = tp / (tp + fp)
    print("pre: " + str(pr))
    rec = tp / (tp + fn)
    print("rec: " + str(rec))
    f1 = 2 * pr * rec / (pr + rec)
    print("f1:" + str(f1))
    return pr, rec, f1

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(top, dis):
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    predictions = []
    targets = []
    for feature in features:
        input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.float)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        # Run prediction for full data
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_mask, segment_ids, label_ids in dataloader: #, desc="Evaluation")
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()

            with torch.no_grad():
                if dis:
                    loss, logits = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)[:2]
                else:
                    loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                     labels=label_ids)[:2]
            sigmoid = torch.nn.Sigmoid()
            logits = sigmoid(logits)

            b_l = input_ids.size(0)

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

        result = {'eval_loss': eval_loss}

        # logger.info("***** Eval results *****")
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
            # writer.write("%s = %s\n" % (key, str(result[key])))

    return targets, predictions


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    args = set_args()

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    paths = [
        # 'checkpoints/para_10000_200_Evaluation_epoch2_ckpt_loss0.08405735123841489.bin',
        # 'checkpoints/para_10000_500_Evaluation_epoch2_ckpt_loss0.08002149834353851.bin',
        # 'checkpoints/para_20000_200_Evaluation_epoch2_ckpt_loss0.07979662920210938.bin',
        # 'checkpoints/para_50000_200_Evaluation_epoch0_ckpt_loss0.08063447918765797.bin',
        # 'checkpoints/para_100000_1000_Evaluation_epoch2_step4000_ckpt_loss0.0754045086341767.bin',
        # 'checkpoints/Evaluation_epoch0_step2500_ckpt_loss0.07643189104053259.bin',
        # 'checkpoints/Evaluation_epoch2_step8500_ckpt_loss0.07637711218634341.bin',
        # 'checkpoints/para_200000_1000_Evaluation_epoch2_step10500_ckpt_loss0.07588262536815593.bin',
        # 'checkpoints/para_200000_1000_1e-5_Evaluation_epoch2_step10000_ckpt_loss0.07299918155307619.bin',
        'para_200000_1000_1e-5_Evaluation_epoch2_step10500_ckpt_loss0.07275297016305225.bin',
        # '0_200000_1000_Evaluation_epoch2_step7642_ckpt_loss0.0734857055758053.bin',
        # '0_200000_1000_Evaluation_epoch2_step7392_ckpt_loss0.0743735213539551.bin',
        '0_200000_1000_Evaluation_epoch1_step6321_ckpt_loss0.074602185646559.bin',
        '0_200000_1000_Evaluation_epoch1_step5571_ckpt_loss0.07488472206903062.bin',
        '1_200000_1000_Evaluation_epoch2_step5250_ckpt_loss0.07530771999424973.bin'
    ]
    dis = False
    for path in paths:
        # Load a trained model that you have fine-tuned
        print(path)
        model_state_dict = torch.load('checkpoints/'+path) #args.ckpt_
        if path[0] == '1':
            dis = True
            args.bert_model = 'distilbert-base-multilingual-cased'
        if path[0] == 'p':
            args.bert_model = 'bert-base-chinese'
        if path[0] == '0':
            args.bert_model = 'bert-base-multilingual-cased'

        if dis:
            tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            model = DistilBertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=1)
        else:
            model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=1)
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        # model = torch.load(args.premodel_path)
        model.cuda()
        model = torch.nn.DataParallel(model)


        processor = DataProcessor()
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.dev_data, args.test_num)
        features = convert_examples_to_features_test(
            examples, label_list, args.max_seq_length, tokenizer, verbose=False)


        ref, res = evaluate(1,dis)
        result(ref, res)
