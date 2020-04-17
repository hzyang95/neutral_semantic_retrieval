import argparse

# python trrr.py --task docretrifull

def set_args():
    parser = argparse.ArgumentParser(description="PyTorch")
    ''' load data and save model'''
    parser.add_argument("--device", type=str, default='1,6,7',
                        help="use GPU")
    parser.add_argument('--raw', action='store_true',default=True,
                        help='是否先做tokenize')
    # parser.add_argument("--train_data", type=str, default='../data/valid.441.2.5.json',
    #                     help="location of dataset")
    # parser.add_argument("--dev_data", type=str, default='../data/valid.441.2.5.json',
    #                     help="location of dataset")
    parser.add_argument("--train_data", type=str, default='../data/data_for_graph_train.v1.30000.143127.2.8.41.json',
                        help="location of dataset")
    parser.add_argument("--dev_data", type=str, default='../data/data_for_graph_valid.2000.10115.3.7.41.json',
                        help="location of dataset")
    parser.add_argument("--test_data", type=str, default='../data/sent_valid.10000.23421.3.8.41.json',
                        help="location of dataset")

    parser.add_argument('--sent_sum_way',
                        type=str, default="avg",
                        help="how to get sentence embedding from bert output")

    # distilbert-base-multilingual-cased
    parser.add_argument("--model", type=str, default="hfl/rbt3",
                        help="type of model to use for Stance Project")

    parser.add_argument("--save", type=str, default='best/sentmixaversplit_final_32_hfl_rbt3_2_7527_0084_4698',
                        help="path to save the model")

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
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")

    parser.add_argument("--test_batch_size", type=int, default=8,
                        help="batch size")

    parser.add_argument("--eval_step", type=int, default=1000,
                        help="eval_step")
    parser.add_argument("--log_step", type=int, default=100,
                        help="log_step")


    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate")


    parser.add_argument("--max_seq_length", type=int, default=500,
                        help="max time step of answer sequence")
    parser.add_argument("--max_ques_length", type=int, default=50,
                        help="max time step of question sequence")

    ''' test purpose'''
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="use CUDA")

    parser.add_argument("--is_test", action="store_true",
                        help="flag for training model or only test")

    parser.add_argument("--out_dir", type=str, default="data/",
                        help="directory for output pickles")

    parser.add_argument("--task", type=str, default='sentmixaversplit',
                        help="use which part of data for training and test")

    parser.add_argument("--proceed", action="store_true",
                        help="flag for continue training on current model")


    args = parser.parse_args()
    return args
