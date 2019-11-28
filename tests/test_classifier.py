import sys
import os


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from neutral_semantic_retrieval.model import Classifier  # noqa


def test_classifier():
    cls = Classifier(os.path.join(CUR_PATH, '../conf/config.yaml'))
    cls.train()

    text = 'M16 步枪'
    assert cls.predict(text) == 'guns'

    texts = ['M16 步枪', '北斗 卫星']
    assert cls.predict(texts) == ['guns', 'spaceship']

    text = '云孚 科技'
    assert cls.predict(text) == 'other'
