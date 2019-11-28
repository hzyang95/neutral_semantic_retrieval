import fasttext
import logging
from .utils import parse_config

logger = logging.getLogger(__name__)


class Classifier(object):
    def __init__(self, conf_file):
        """
        Args:
            conf_file: 配置文件，包括训练数据、模型位置等文件路径配置以及模型参数配置。
        """
        logger.info('parse config from: {}'.format(conf_file))
        conf = parse_config(conf_file)
        data_conf, params_conf = conf['path'], conf['params']
        self.train_data = data_conf['train_data']
        self.model_file = data_conf['model_file']
        self.epoch = params_conf['epoch']
        self.label_prefix = params_conf['label_prefix']
        self.thread = params_conf['thread']
        self.neg = params_conf['neg']

    def train(self):
        """
        Returns:

        """
        logger.info('begin training, model_file: {}'.format(self.model_file))
        model = fasttext.supervised(
            self.train_data, self.model_file, label_prefix=self.label_prefix,
            epoch=self.epoch, thread=self.thread, neg=self.neg)
        logger.info('end training')
        logger.info('model saved to: {}'.format(self.model_file))
        return model

    def _remove_label_prefix(self, label):
        return label[len(self.label_prefix):]

    def predict(self, text):
        """
        Args:
            text: 待预测的文本或文本列表
        Returns:
            预测的类别或类别列表
        """
        logger.info('model load from : {}'.format(self.model_file))
        model = fasttext.load_model('{}.bin'.format(self.model_file))
        if isinstance(text, list):
            return [self._remove_label_prefix(i[0]) for i in model.predict(text)]
        elif isinstance(text, str):
            return self._remove_label_prefix(model.predict([text])[0][0])
        else:
            raise Exception("Invalid text type: {}!".format(type(text)))
