from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import pandas
from tqdm import tqdm
from config import set_args
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import Counter
from transformers import BertTokenizer,DistilBertTokenizer
from transformers import BertForSequenceClassification,DistilBertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(torch.cuda.device_count())
class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self, data_dir, top):
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "hotpot_ss_train.csv")))
        # train_path = os.path.join(data_dir, "hotpot_ss_train.csv")
        train_path = data_dir
        return self._create_examples_train(
            train_path, set_type='train', top=top)

    def get_dev_examples(self, data_dir, top):
        # dev_path = os.path.join(data_dir, "hotpot_ss_dev.csv")
        return self._create_examples_dev(
            data_dir, set_type='dev', top=top)

    def get_test_examples(self, data_dir, top):
        # dev_path = os.path.join(data_dir, "hotpot_ss_dev.csv")
        return self.create_examples_test(
            data_dir, set_type='test', top=top)

    def get_labels(self):
        return [False, True]

    def _create_examples_train(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        for i, line in enumerate(file[:top]):
            row = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            text_a = row['question']
            # text_b = '{} {}'.format(row['context'], row['title'])
            text_b = '{}'.format(row['text'])
            label = row['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # for (i, row) in df.iterrows():
        #     guid = "%s-%s" % (set_type, i)
        #     text_a = row['question']
        #     # text_b = '{} {}'.format(row['context'], row['title'])
        #     text_b = '{}'.format(row['context'])
        #     label = row['label']
        #     examples.append(
        #         InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_dev(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        num = 0
        for i, line in enumerate(file[:top]):
            row = json.loads(line)
            for item in row['sample']:
                num += 1
                guid = "%s-%s" % (set_type, num)
                text_a = item['question']
                # text_b = '{} {}'.format(row['context'], row['title'])
                text_b = '{}'.format(item['text'])
                label = item['label']
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def create_examples_test(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        for i, line in enumerate(file[1000:1000+top]):
            row = json.loads(line)
            samples = []
            for item in row['sample']:
                guid = "%s-%s" % (set_type, i)
                text_a = item['question']
                # text_b = '{} {}'.format(row['context'], row['title'])
                text_b = '{}'.format(item['text'])
                label = item['label']
                samples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            examples.append(samples)
        return examples


def convert_examples_to_features(exampless, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    res = []
    for examples in exampless:
        features = []
        for (ex_index, example) in enumerate((examples)):
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # Feature ids
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a ) + 2) + [1] * (len(tokens_b) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Mask and Paddings
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5 and verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
        res.append(features)
    return res


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


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(top):
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


def get_dev_paras(data, pred_score, output_path):

    logits = np.array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]

    Paragraphs = dict()
    cur_ptr = 0

    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr
        all_paras = []
        selected_paras = []
        while cur_ptr < tem_ptr + len(case['context']):
            score = pred_score.ix[cur_ptr, 'prob']
            all_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            if score >= 0.05:  # 0.05
                selected_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            cur_ptr += 1
        sorted_all_paras = sorted(all_paras, key=lambda x: x[0], reverse=True)
        sorted_selected_paras = sorted(selected_paras, key=lambda x: x[0], reverse=True)
        Paragraphs[key] = [p[1] for p in sorted_selected_paras]
        while len(Paragraphs[key]) < 3:
            if len(Paragraphs[key]) == len(all_paras):
                break
            Paragraphs[key].append(sorted_all_paras[len(Paragraphs[key])][1])

    Selected_paras_num = [len(Paragraphs[key]) for key in Paragraphs]
    print("Selected Paras Num:", Counter(Selected_paras_num))

    json.dump(Paragraphs, open(output_path, 'w'))


def get_train_paras(source_data, score, output_path):
    # + Negative Sample.
    logits = np.array([score['logits0'], score['logits1']]).transpose()
    score['prob'] = softmax(logits)[:, 1]
    score = np.array(score['prob'])
    Paragraphs = dict()
    ptr = 0
    for case in tqdm(source_data):
        key = case['_id']
        Paragraphs[key] = []
        para_ids = []
        gold = set([para[0] for para in case['supporting_facts']])

        for i, para in enumerate(case['context']):
            if para[0] in gold:
                Paragraphs[key].append(para)
                para_ids.append(i)

        tem_score = score[ptr:ptr + len(case['context'])]
        ptr += len(case['context'])
        sorted_id = sorted(range(len(tem_score)), key=lambda k: tem_score[k], reverse=True)

        for i in sorted_id:
            if i not in para_ids:
                Paragraphs[key].append(case['context'][i])
                break

    json.dump(Paragraphs, open(output_path, 'w'))


def get_dataframe(file_path):
    source_data = json.load(open(file_path, 'r'))
    sentence_pair_list = []
    for case in source_data:
        for para in case['context']:
            pair_dict = dict()
            pair_dict['label'] = 0
            pair_dict['title'] = para[0]
            pair_dict['context'] = " ".join(para[1])
            pair_dict['question'] = case['question']
            sentence_pair_list.append(pair_dict)

    return source_data, pandas.DataFrame(sentence_pair_list)


if __name__ == "__main__":
    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    paths = [
        # 'checkpoints/para_10000_200_Evaluation_epoch2_ckpt_loss0.08405735123841489.bin',
        # 'checkpoints/para_10000_500_Evaluation_epoch2_ckpt_loss0.08002149834353851.bin',
        # 'checkpoints/para_20000_200_Evaluation_epoch2_ckpt_loss0.07979662920210938.bin',
        # 'checkpoints/para_50000_200_Evaluation_epoch0_ckpt_loss0.08063447918765797.bin',
        # 'checkpoints/para_100000_1000_Evaluation_epoch2_step4000_ckpt_loss0.0754045086341767.bin',
        # 'checkpoints/Evaluation_epoch0_step2500_ckpt_loss0.07643189104053259.bin',
        # 'checkpoints/Evaluation_epoch2_step8500_ckpt_loss0.07637711218634341.bin',
        'checkpoints/para_200000_1000_Evaluation_epoch2_step10500_ckpt_loss0.07588262536815593.bin',
        # 'checkpoints/para_200000_1000_1e-5_Evaluation_epoch2_step10000_ckpt_loss0.07299918155307619.bin',
        'checkpoints/para_200000_1000_1e-5_Evaluation_epoch2_step10500_ckpt_loss0.07275297016305225.bin'
    ]
    for path in paths:
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(path) #args.ckpt_
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=1)
        # model = DistilBertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=1)
        # model = torch.load(args.premodel_path)
        model.cuda()
        model = torch.nn.DataParallel(model)


        processor = DataProcessor()
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.dev_data, args.test_num)
        features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, verbose=False)


        ref, res = evaluate(1)
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
