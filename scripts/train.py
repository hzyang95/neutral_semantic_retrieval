import sys
import os
import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from neu_sem_retrieval.trainer import Trainer  # noqa
# from classifier.trainer import Trainer  # noqa

if __name__ == '__main__':
    conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
    clf = Trainer(conf_file)
    clf.train()
