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
        # for (i, row) in df.iterrows():
        #     guid = "%s-%s" % (set_type, i)
        #     question = row['question']
        #     # answer = '{} {}'.format(row['context'], row['title'])
        #     answer = '{}'.format(row['context'])
        #     label = row['label']
        #     examples.append(
        #         InputExample(guid=guid, question=question, answer=answer, label=label))
        return examples

    # def _create_examples_dev(self, path, set_type, top):
    #         examples = []
    #         with open(path, 'r', encoding='utf-8') as f:
    #             file = f.readlines()
    #         num = 0
    #         for i, line in enumerate(file[:top]):
    #             row = json.loads(line)
    #             for item in row['sample']:
    #                 num += 1
    #                 guid = "%s-%s" % (set_type, num)
    #                 question = item['question']
    #                 # answer = '{} {}'.format(row['context'], row['title'])
    #                 answer = '{}'.format(item['text'])
    #                 label = item['label']
    #                 examples.append(
    #                     InputExample(guid=guid, question=question, answer=answer, label=label))
    #
    #         return examples
    def _create_examples_dev(self, path, set_type, top):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        for i, line in enumerate(file[:]):
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
        for i, line in enumerate(file[top:top + top]):
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
