import time
import datetime as dt

import tornado.gen
import tornado.httpserver
import tornado.ioloop
import tornado.web
import json

class SleepHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.num = '123'

    @tornado.gen.coroutine
    def post(self):
        print(self.request.body)
        data = json.loads(self.request.body)
        # time.sleep(5)
        # yield tornado.gen.sleep(3)
        self.write({'res': data['num']})


if __name__ == "__main__":
    app = tornado.web.Application(
        [
            (r"/sleep", SleepHandler)
        ],
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(56789)
    print('server start')
    tornado.ioloop.IOLoop.instance().start()
