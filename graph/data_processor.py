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

    def __init__(self, input_ids, input_mask, segment_ids, sp_label, sent_start, sent_end, wd_edges, graph_mask,
                 sent_num):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sp_label = sp_label
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.wd_edges = wd_edges
        self.graph_mask = graph_mask
        self.sent_num = sent_num


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


class DataProcessor(object):

    def get_labels(self):
        return [False, True]

    def create_examples(self, path, set_type, test=False, is_train=True, sent=False):
        examples = []
        print(sent)
        with open(path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        if test:
            file = file[:800]
        print(len(file))
        _aver = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        nnnn = 0
        neg = []
        neg_long = []
        all_num=0
        # lb0 = 0
        for i, line in enumerate(tqdm(file[:])):
            row = json.loads(line)
            guid = "%s-%s" % (set_type, i)
            question = row['question']
            content = []
            label = []
            ll = 0
            # if row['top'] == 0:
            #     lb0 += 1
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
                # text_b = '{} {}'.format(row['context'], row['title'])
                if (not sent and (ll + lll > 400 or len(content) + 1 >= 30)) or(sent and(len(content) + 1 >= 50)):
                # if ll + lll > 400 or len(content) + 1 >= 30:
                    # print(len(content))
                    if len(label) == 0:
                        continue
                    assert len(content) == len(label)
                    # ssss = 0
                    # for i in label:
                    #     if i == 1:
                    #         ssss = 1
                    #         break
                    # if ssss == 0:
                    #     nnnn += 1
                    len_of_cont = len(content)
                    _aver[len_of_cont // 10] += 1
                    # if lll // 10 < 3:
                    #     _aver[lll // 10] += 1
                    # else:
                    #     _aver[3] += 1
                    #     # print(lll)
                    #     # print(content)
                    #     continue
                    all_num +=len_of_cont
                    examples.append(InputExample(guid=guid, text_a=question, text_b=content, label=label))
                    content = []
                    label = []
                    ll = 0

                ll += lll
                content.append(item['text'])
                label.append(item['label'])
            # print(len(content))
            if len(label) == 0:
                continue
            assert len(content) == len(label)
            len_of_cont = len(content)
            _aver[len_of_cont // 10] += 1
            all_num += len_of_cont
            # if lll // 10 < 3:
            #     _aver[lll // 10] += 1
            # else:
            #     _aver[3] += 1
            #     continue
            #     # print(ll)
            #     # print(content)
            # if ll // 100 < 5:
            #     _aver[ll // 100] += 1
            # else:
            #     _aver[5] += 1
            #     continue
            # ssss = 0
            # for i in label:
            #     if i == 1:
            #         ssss = 1
            #         break
            # if ssss == 0:
            #     nnnn += 1
            # print(question)
            # print(content)
            # print(row['sample'])
            # print(row['top'])
            # print(label)
            # if len('.'.join(content))>300:
            #     continue
            # if ll > 300:
            #     continue
            assert len(label) == len(content)
            examples.append(InputExample(guid=guid, text_a=question, text_b=content, label=label))
        print(_aver)
        print(all_num)
        # print(len(neg))
        # # print('\n'.join(neg[:10]))
        # print(len(neg_long))
        # # print('\n'.join(neg_long[:10]))
        # # print(nnnn)
        # # print(lb0)
        print(len(examples))
        logging.info(str(set_type) + str(len(examples)))
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

        sent_num = len(sent_start)

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
                        # if abs(s1i - s2i) <= 3:
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
                          graph_mask=graph_mask,
                          sent_num=sent_num
                          ))
    return features


def convert_examples_to_features_sent(examples, label_list, max_query_length, max_seq_length, tokenizer, is_train=True):
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
        input_ids = []
        input_mask = []
        segment_ids = []
        cur_pos = 0
        for si, sent in enumerate(example.text_b):
            sent_start.append(cur_pos)
            cur_tokens = tokenizer.tokenize(sent)
            _truncate_seq_pair(query_tokens, cur_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + cur_tokens + ["[SEP]"]
            sent_segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(cur_tokens) + 1)
            sent_input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Mask and Paddings
            sent_input_mask = [1] * len(sent_input_ids)
            sent_padding = [0] * (max_seq_length - len(sent_input_ids))

            sent_input_ids += sent_padding
            sent_input_mask += sent_padding
            sent_segment_ids += sent_padding

            assert len(sent_input_ids) == max_seq_length
            assert len(sent_input_mask) == max_seq_length
            assert len(sent_segment_ids) == max_seq_length

            graph_mask.append(1)
            if is_train:
                sp_label.append(int(example.label[si]))
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            segment_ids.append(sent_segment_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        sent_num = len(input_ids)

        # print(sent_tokens)
        # print(sent_start)
        # print(sent_end)
        # print(graph_mask)
        # print(sp_label)

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
                        # if abs(s1i - s2i) <= 3:
                        wd_edge_list.append([s1i, s2i])
            return wd_edge_list

        wd_edges = wd_edge_list(input_ids)
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
        #     # logger.info("tokens: %s" % " ".join(
        #     #             #     [str(x) for x in tokens]))
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
                          graph_mask=graph_mask,
                          sent_num=sent_num
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


if __name__ == "__main__":
    dataset = [
        'data_for_graph_train.v1.30000.143127.2.8.41.json',
        'data_for_graph_valid.2000.10115.3.7.41.json',
        'data_for_graph_test.3000.13848.3.9.40.json'
    ]
    dt = DataProcessor()
    for i in dataset:
        # dt.create_examples('../data/' + i, 'test', False)
        dt.create_examples('../../../data/cips-sougou/nottruth/' + i, 'test', False,True,sent=False)
