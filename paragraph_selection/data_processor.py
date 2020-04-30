import json


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


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

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
            # answer = '{} {}'.format(row['context'], row['title'])
            text_b = '{}'.format(row['text'])
            label = row['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_dev(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        for i, line in enumerate(file[:top]):
            row = json.loads(line)
            samples = []
            for item in row['sample']:
                guid = "%s-%s" % (set_type, i)
                text_a = item['question']
                # answer = '{} {}'.format(row['context'], row['title'])
                text_b = '{}'.format(item['text'])
                label = item['label']
                samples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            examples.append(samples)
        return examples

    def create_examples_test(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        print(len(file))
        print(top)
        _aver = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        nnnn = 0
        neg = []
        neg_long = []
        # lb0 = 0
        for i, line in enumerate(file):
            row = json.loads(line)
            samples = []
            question = row['question']
            ll=0
            for item in row['sample']:
                lll = len(item['text'])
                # print(ll)
                if check_contain_chinese(item['text']) is False:
                    continue
                # if lll // 100 < 4:
                #     _aver[lll // 100] += 1
                # else:
                #     _aver[4] += 1
                #     # print(lll)
                #     # print(item['text'])
                if lll < 5:
                    neg.append(question + ':    ' + item['text'])
                    continue
                if lll > 200:
                    # if lll < 300:
                    neg_long.append(question + ':   ' + item['text'])
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = item['question']
                # answer = '{} {}'.format(row['context'], row['title'])
                text_b = '{}'.format(item['text'])
                ll += len(item['text'])
                label = item['label']
                samples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            # if ll>300:
            #     continue
            if len(samples) == 0:
                continue
            lll = len(samples)
            if lll // 10 < 3:
                _aver[lll // 10] += 1
            else:
                _aver[3] += 1
                # print(ll)
                # print(samples)
            examples.append(samples)
        print(_aver)
        print(len(neg))
        # print('\n'.join(neg[:10]))
        print(len(neg_long))
        # print('\n'.join(neg_long[:10]))
        # print(nnnn)
        # print(lb0)
        print(len(examples))
        return examples


def convert_examples_to_features_test(exampless, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    res = []
    for examples in exampless:
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # Feature ids
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
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
            # if ex_index < 5 and verbose:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        res.append(features)
    return res

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # Feature ids
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
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
        # if ex_index < 5 and verbose:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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

if __name__ == "__main__":
    dataset = [
        'data_for_graph_train.v1.30000.143127.2.8.41.json',
        'data_for_graph_valid.2000.10115.3.7.41.json',
        'data_for_graph_test.3000.13848.3.9.40.json'
    ]
    dt = DataProcessor()
    for i in range(3):
        dt.create_examples_test('../data/'+dataset[i], 'test', False)

