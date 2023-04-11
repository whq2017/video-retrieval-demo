from flask import Flask

from flask_cors import CORS


app = Flask(__name__)
cors = CORS()
cors.init_app(app=app, resources={
    r'/api/*': {"origins": "*"}
})


def run(web_config):
    # 注册全局属性信息
    app.config['search'] = web_config.search
    app.config['dataset'] = web_config.dataset
    # if web_config.run:
    app.run(port=web_config.port, debug=web_config.debug)


