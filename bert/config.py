import argparse

# python trrr.py --task docretrifull

def set_args():
    parser = argparse.ArgumentParser(description="PyTorch RoomConditonal for Stance Project")
    ''' load data and save model'''
    parser.add_argument("--input", type=str, default="mixaver",
                        help="location of dataset")


    parser.add_argument("--model", type=str, default="hfl/rbt3",
                        help="type of model to use for Stance Project")

    parser.add_argument("--saved", type=str, default=None,
                        help="path to save the model")

    parser.add_argument("--save", type=str, default='best',
                        help="path to save the model")

    parser.add_argument("--num_labels", type=int, default=2,
                        help="number of label")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of training epoch")
    parser.add_argument("--batch_size", type=int, default=140,
                        help="batch size")

    parser.add_argument("--test_batch_size", type=int, default=70,
                        help="batch size")

    parser.add_argument("--eval_step", type=int, default=1000,
                        help="eval_step")
    parser.add_argument("--log_step", type=int, default=100,
                        help="log_step")


    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate")
    parser.add_argument("--ans_len", type=int, default=500,
                        help="max time step of answer sequence")
    parser.add_argument("--ask_len", type=int, default=25,
                        help="max time step of question sequence")

    ''' test purpose'''
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true",default=True,
                        help="use CUDA")
    parser.add_argument("--device", type=str, default='1,2,3,4',
                        help="use GPU")
    parser.add_argument("--is_test", action="store_true",
                        help="flag for training model or only test")

    parser.add_argument("--out_dir", type=str, default="data/vec/",
                        help="directory for output pickles")

    parser.add_argument("--task", type=str, default='sentmixaversplit',
                        help="use which part of data for training and test")

    parser.add_argument("--proceed", action="store_true",
                        help="flag for continue training on current model")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()
    return args
