import sys
import os


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from neutral_semantic_retrieval.service.serve import model_serve

if __name__ == '__main__':
    conf_file = os.path.join(CUR_PATH, '../conf/config.yaml')
    model_serve(conf_file)
