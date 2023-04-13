from flask import Flask, render_template

from flask_cors import CORS


app = Flask(__name__,
            static_folder='./vue/dist/assets',  # 设置静态文件夹目录
            template_folder="./vue/dist")  # 设置vue编译输出目录dist文件夹，为Flask模板文件目录

cors = CORS()
cors.init_app(app=app, resources={
    r'/api/*': {"origins": "*"},
    r'/api/video/*': {"origins": "*"}
})


@app.route('/')
def index():
    return render_template('index.html', name='index')


@app.route('/favicon.ico')
def ico():
    return render_template('favicon.ico', name='favicon')


def run(web_config):
    # 注册全局属性信息
    app.config['search'] = web_config.search
    app.config['dataset'] = web_config.dataset
    # if web_config.run:
    app.run(port=web_config.port, debug=web_config.debug)


