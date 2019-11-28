import json
import logging

import tornado.ioloop
import tornado.web

from neutral_semantic_retrieval.utils import parse_config
from neutral_semantic_retrieval.model import Classifier

logger = logging.getLogger(__name__)


class PredictHandler(tornado.web.RequestHandler):
    model = None

    def initialize(self, conf_file):
        if not PredictHandler.model:
            logger.info('init model in PredictHandler')
            PredictHandler.model = Classifier(conf_file)

    def post(self):
        data = json.loads(self.request.body)
        text = data['text']
        prediction = self.model.predict(text)
        logger.info('do prediction: {}'.format(prediction))
        self.finish({'result': prediction})


def make_app(conf_file):
    app = tornado.web.Application([
        (r"/predict", PredictHandler, {
            "conf_file": conf_file
        }),
    ])
    return app


def model_serve(conf_file):
    conf = parse_config(conf_file)
    port = conf['serve']['port']
    n_process = conf['serve']['n_process']
    logger.info('Starting %s processes' % n_process)
    app = make_app(conf_file)
    server = tornado.httpserver.HTTPServer(app)
    server.bind(port)
    server.start(n_process)
    tornado.ioloop.IOLoop.instance().start()
