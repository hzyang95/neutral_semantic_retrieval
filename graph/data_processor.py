import json
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, sp_label, sent_start, sent_end, wd_edges, graph_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sp_label = sp_label
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.wd_edges = wd_edges
        self.graph_mask = graph_mask


class DataProcessor(object):

    def get_labels(self):
        return [False, True]

    def create_examples(self, path, set_type, test=False, is_train=True):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        if test:
            file = file[:8000]
        for i, line in enumerate(tqdm(file[:])):
            row = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            question = row['question']
            content = []
            label = []
            for item in row['sample']:
                guid = "%s-%s" % (set_type, i)
                # text_b = '{} {}'.format(row['context'], row['title'])
                content.append(item['text'])
                label.append(item['label'])
            print(len(content))
            assert len(content) == len(label)
            if len('.'.join(content))>300:
                continue
            examples.append(InputExample(guid=guid, text_a=question, text_b=content, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_query_length, max_seq_length, tokenizer, is_train=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    for (ex_index, example) in enumerate(tqdm(examples)):

        graph_mask = []
        sent_start = []
        sent_end = []
        sp_label = None

        if is_train:
            sp_label = []

        query_tokens = tokenizer.tokenize(example.text_a)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        sent_tokens = []
        sent_text = " ".join([sent for sent in example.text_b]).strip()
        # print(sent_text)

        cur_pos = 0
        for si, sent in enumerate(example.text_b):
            sent_start.append(cur_pos)
            cur_tokens = tokenizer.tokenize(sent)
            sent_tokens.extend(cur_tokens)
            sent_end.append(cur_pos + len(cur_tokens))
            cur_pos += len(cur_tokens)
            graph_mask.append(1)
            if is_train:
                sp_label.append(int(example.label[si]))

        # print(sent_tokens)
        # print(sent_start)
        # print(sent_end)
        # print(graph_mask)
        # print(sp_label)

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        cls_index = 0
        # print(len(tokens))
        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        # print(len(tokens))
        # SEP token
        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)
        # print(len(tokens))
        for token in sent_tokens:
            tokens.append(token)
            segment_ids.append(1)
        # print(len(tokens))
        # SEP token
        tokens.append(tokenizer.sep_token)
        segment_ids.append(1)
        # print(len(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]

        for si in range(len(sent_start)):
            sent_start[si] += len(query_tokens) + 2
            sent_end[si] += len(query_tokens) + 2

        sent_start = [i for i in sent_start if i < max_seq_length]
        sent_end = sent_end[:len(sent_start)]

        # print(sent_start)
        # print(sent_end)
        #
        # print(len(input_ids))
        # print(max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(sent_start) == len(sent_end)

        def wd_edge_list(data):
            """with-document edge list

            Arguments:
                data {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            wd_edge_list = []
            for s1i, s1 in enumerate(data):  # even for doc title, odd for doc sents
                for s2i, s2 in enumerate(data):
                    if s1i != s2i:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        wd_edges = wd_edge_list(sent_start)
        # print(wd_edges)

        # content = tokenizer.tokenize(sent_text)
        # _truncate_seq_pair(query_tokens, content, max_seq_length - 3)
        #
        # # Feature ids
        # tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + content + ["[SEP]"]
        # segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(content) + 1)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #
        # # Mask and Paddings
        # input_mask = [1] * len(input_ids)
        # padding = [0] * (max_seq_length - len(input_ids))
        #
        # input_ids += padding
        # input_mask += padding
        # segment_ids += padding
        #
        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        #
        # if is_train:
        #     label_id = label_map[example.label[0]]
        # else:
        #     label_id = None

        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s" % " ".join([str(x) for x in sp_label]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sp_label=sp_label,
                          sent_start=sent_start,
                          sent_end=sent_end,
                          wd_edges=wd_edges,
                          graph_mask=graph_mask
                          ))
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
