import argparse


# python trrr.py --task docretrifull

def set_args():
    parser = argparse.ArgumentParser(description="PyTorch")
    ''' load data and save model'''

    parser.add_argument("--task", type=str, default='sentmixaversplit',
                        help="use which part of data for training and test")


    parser.add_argument("--device", type=str, default='3',
                        help="use GPU")

    parser.add_argument("--testbatch", action='store_true', default=False,
                        help="testbatch mode")

    parser.add_argument("--sent", action='store_true',default=True,
                        help="sent mode")

    parser.add_argument("--full_pas", action='store_true', #default=True,
                        help="sent mode")

    parser.add_argument("--wdedge", action='store_true',  default=True,
                        help="wdedge")
    parser.add_argument("--quesedge", action='store_true',  # default=True,
                        help="quesedge")
    parser.add_argument("--adedge", action='store_true',  # default=True,
                        help="adedge")

    parser.add_argument("--is_test", action="store_true", #default=True,
                        help="flag for training model or only test")

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="select threshold")

    parser.add_argument('--raw', action='store_true',  default=True,
                        help='是否先做tokenize')

    parser.add_argument('--dice', action='store_true', #default=True,
                        help='dice_loss')
    # parser.add_argument("--train_data", type=str, default='../data/valid.441.2.5.json',
    #                     help="location of dataset")
    # parser.add_argument("--dev_data", type=str, default='../data/valid.441.2.5.json',
    #                     help="location of dataset")


    parser.add_argument("--train_data", type=str, default='../data/data_for_graph_train.v1.30000.143127.2.8.41.json',
                        help="location of dataset")
    parser.add_argument("--dev_data", type=str, default='../data/data_for_graph_valid.2000.10115.3.7.41.json',
                        help="location of dataset")
    parser.add_argument("--test_data", type=str, default='../data/data_for_graph_test.3000.13848.3.9.40.json',
                        help="location of dataset")

    parser.add_argument("--train_data_cmrc", type=str, default='../data/cmrc2018__train.10142.10.21.41.json',
                        help="location of dataset")
    # cmrc2018__train.10142.1.12.41
    # cmrc2018__train.10142.1.2.41.json
    # cmrc2018__train.10142.10.21.41.json
    parser.add_argument("--dev_data_cmrc", type=str, default='../data/cmrc2018__dev.1548.1.12.40.json',
                        help="location of dataset")
    parser.add_argument("--test_data_cmrc", type=str, default='../data/cmrc2018__test.1671.1.13.38.json',
                        help="location of dataset")

    parser.add_argument("--train_data_full", type=str, default='../data/ques_data_for_graph_train.v1.30000.27127.2.8.41.json',
                        help="location of dataset")
    parser.add_argument("--dev_data_full", type=str, default='../data/ques_data_for_graph_valid.2000.10115.1809.3.7.41.json',
                        help="location of dataset")
    parser.add_argument("--test_data_full", type=str, default='../data/ques_data_for_graph_test.3000.13848.2621.3.9.40.json',
                        help="location of dataset")

    # ../../../data/cips-sougou/nottruth/
    # sent_valid.10000.23421.3.8.41.json
    # data_for_graph_test.3000.13848.3.9.40.json
    # ques_data_for_graph_train.v1.30000.27127.2.8.41.json
    # ques_data_for_graph_valid.2000.10115.1809.3.7.41.json
    # ques_data_for_graph_test.3000.13848.2621.3.9.40.json
    parser.add_argument('--sent_sum_way',
                        type=str, default="aver",
                        help="how to get sentence embedding from bert output")

    # distilbert-base-multilingual-cased
    # hfl/rbt3
    parser.add_argument("--model", type=str, default="hfl/rbt3",
                        help="type of model to use for Stance Project")

    model = ['sentmixaversplit_final_32_distilbert-base-multilingual-cased_6_16000_0021_7199__short300',
             'sentmixaversplit_final_32_hfl_rbt3_2_7527_0084_4698_short300',
             'sentmixaversplit_final_32_hfl_rbt3_4_12730_0043_7416_newshort300',
             'sentmixaversplit_final_32_hfl_rbt3_2_7638_0045_7414_short300_l3',
             'sentmixaversplit_96_hfl_rbt3_2_6333_0033_7269',
             'sentmixaversplit_final_32_hfl_rbt3_4_31655_0084_6791_sent_ori']

    parser.add_argument("--save", type=str,
                        default='best/'+model[5],
                        help="path to save the model")
    # sentmixaversplit_final_32_distilbert-base-multilingual-cased_6_16000_0021_7199__short300
    # sentmixaversplit_final_32_hfl_rbt3_2_7527_0084_4698_short300
    #
    parser.add_argument("--num_labels", type=int, default=2,
                        help="number of label")

    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--warmup_steps", default=0.1, type=float,
                        help="Linear warmup over warmup_steps. Actually warmup ratio")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--epochs", type=int, default=25,
                        help="number of training epoch")

    parser.add_argument("--early_stop", type=int, default=15,
                        help="number of training epoch")

    parser.add_argument("--batch_size", type=int, default=6,
                        help="batch size")

    parser.add_argument("--test_batch_size", type=int, default=4,
                        help="batch size")

    parser.add_argument("--eval_step", type=int, default=50,
                        help="eval_step")
    parser.add_argument("--log_step", type=int, default=10,
                        help="log_step")

    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate")

    parser.add_argument("--max_seq_length", type=int, default=150,
                        help="max time step of answer sequence")
    parser.add_argument("--max_ques_length", type=int, default=50,
                        help="max time step of question sequence")

    ''' test purpose'''
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="use CUDA")

    parser.add_argument("--out_dir", type=str, default="data/",
                        help="directory for output pickles")

    parser.add_argument("--proceed", action="store_true",
                        help="flag for continue training on current model")

    args = parser.parse_args()
    return args
