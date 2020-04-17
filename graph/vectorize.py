import numpy as np
import os
import time
import sys

from tqdm import tqdm

sys.path.append('../')
from utils.file_utils import data_to_pickle, pickle_to_data, write_list2file, write_lol2file
from utils.tools import shuffle



from transformers import BertTokenizer, DistilBertTokenizer
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification

np.random.seed(1234567)


################################
# Raw data --> json or pickle
# output file style looks like this:
#     {"training":{
#         "xIndexes":[]
#         "yLabels":[]
#         "xQuestions":[]
#             }
#      "validation": (looks like training)
#      "test": (looks like training)
#      "word2idx":{"_padding":0,"_unk":1, "1st":2, "hello":3, ...}
#      "label2idx"{"-1":0, "0":1, "1":2}
#      "embedding":[ [word0], [word1], [word2], ...] ********************* Can be moved to separate file
#     }
################################

# read segmented corpus into list (sentence in list of words)

def read_file2list(filename):
    with open(filename, 'rb') as f:
        contents = [line.strip().decode() for line in f]
    print("The %s has lines: %d" % (filename, len(contents)))
    return contents

task = 'docretrifulltest'
# task = 'newsimmixaver'
# task = 'sentretri_dis'
# model = 'distilbert-base-multilingual-cased'
model = 'hfl/rbt3'
# sentmixaveronlysingle
# sentmixaverltp
def read_dataset(args):
    """
    Make it easy to generate training and test sets
    1) decide to_read directories
    2) split the data set into training/test/validation
    3) ??? what about cross-validation
    :param args:
    :return:
    """
    data = []  # answer, label, question, remark
    # fns = ["newsimmixaver_train_14341_4802_4695_4844", "newsimmixaver_test_3584_1200_1173_1211"]     # "../data/10k/"
    # fns = ["sentretri_train_1205950_375052_830898_0", "sentretri_test_200016_73508_126508_0"]     # "../data/10k/"
    fns = ["docretrifull_train_1285618_696005_589613",
           "docretrifull_test_45811_26252_19559"]     # "../data/10k/"

    for fn in fns:
        data.append([read_file2list("%s/%s/answers.txt" % (args.in_dir, fn)),
                     read_file2list("%s/%s/questions.txt" % (args.in_dir, fn)),
                     read_file2list("%s/%s/labels.txt" % (args.in_dir, fn))]
                    )

    assert len(data[0][0]) == len(data[0][1]) == len(data[0][2])
    assert len(data[1][0]) == len(data[1][1]) == len(data[1][2])

    # Data follows this order: train, test
    shuffle(data[0], seed=123456)
    test = data[1]
    if args.has_valid:
        train_num = int(len(data[0][0]) * args.portion)
        train = [d[:train_num] for d in data[0]]
        valid = [d[train_num:] for d in data[0]]
    else:
        train = data[0]
        valid = test

    assert len(train[0]) == len(train[1]) == len(train[2])
    assert len(valid[0]) == len(valid[1]) == len(valid[2])
    assert len(test[0]) == len(test[1]) == len(test[2])

    raw_data = {"training": train,
                "validation": valid,
                "test": test}
    return raw_data


def convert_examples_to_features(examples,  max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {'0': 0, '1': 1}
    features = []
    _len = len(examples[0][:20000])
    for i in tqdm(range(_len)):
        ans = examples[0][i]
        ques = examples[1][i]
        label = examples[2][i]
        # print(len(ans))
        ques = tokenizer.tokenize(ques)
        ans = tokenizer.tokenize(ans)
        _truncate_seq_pair(ques, ans, max_seq_length - 3)

        # Feature ids
        tokens = ["[CLS]"] + ques + ["[SEP]"] + ans + ["[SEP]"]
        segment_ids = [0] * (len(ques) + 2) + [1] * (len(ans) + 1)
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

        label_id = label_map[label]

        features.append({'input_ids': input_ids,
                          'input_mask': input_mask,
                          'segment_ids': segment_ids,
                          'label_id': label_id})
    return features

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

def processing(args):
    out_dir = args.out_dir + "/" + str(task)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 1. Split the data, read into defined format
    raw_data = read_dataset(args)
    # print(raw_data['test'][2])

    if args.model[:3] == 'dis':
        tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)
    dataset = {}
    for key in raw_data:
        print('start: {}'.format(key))
        dataset[key] = convert_examples_to_features(raw_data[key],args.max_len,tokenizer)
        # print(dataset)

    # 3. Transform text into indexes
    # datasets = make_datasets(raw_data)

    # 4. Write training materials into pickles
    data_to_pickle(dataset, out_dir + "/features.pkl")
    #
    # print("hello------------------------")
    # tit = time.time()
    # # 5. Pad the training data
    # preload_tvt(datasets, max_lens=[args.sen_max_len, args.ask_max_len],
    #             out_dir=out_dir, emb=args.emb, feat_names=feat_names)
    # print("hello------------------------", (time.time() - tit))
    # test correctness
    # datasets = pickle_to_data(out_dir + "/features.pkl")
    # word2idx = pickle_to_data(out_dir + "/word2idx_" + args.emb + ".pkl")
    #
    # print(len(datasets["test"]))
    # print(datasets["test"][0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vectorize Sogou AnswerStance data")
    ''' May not changed '''
    parser.add_argument("--in_dir", type=str, default="processed",
                        help="directory for input data")
    parser.add_argument("--out_dir", type=str, default="data/vec/",
                        help="directory for output pickles")
    parser.add_argument("--task", type=str, default=task,
                        help="use which part of data for training and test")

    ''' May change '''
    parser.add_argument("--has_valid", default=True, action="store_true",
                        help="whether have 'real' validation data for tuning the model")
    parser.add_argument("--portion", type=float, default=0.9,
                        help="decide portion to spare for training and validation")
    parser.add_argument("--max_len", type=int, default=500,
                        help="max time step of sentence sequence")

    parser.add_argument("--model", type=str, default=model)
    my_args = parser.parse_args()

    processing(my_args)
