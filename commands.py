import argparse

parser = argparse.ArgumentParser(description='-h/--help中的文档描述')

__all__ = ['parser']

parser.add_argument('run', type=str, default='pipe',
                    help='启动的方式，可以选择的值：`web`，`pipe`（默认）\n'
                         '`pipe`方式只支持连接本地Milvus')

# ##################### Dataset Config ####################
parser.add_argument('-g', '--create_gif', action='store_true', default=False,
                    help='加载数据库信息是否为视频创建GIF图片')
parser.add_argument('--num', type=int, default=16,
                    help='GIF图片抽取的帧数')
parser.add_argument('--path', type=str, default='',
                    help='指定数据库所在位置，或者下载位置')

# ##################### Milvus Config #####################
parser.add_argument('-r', '--remote', default=False,
                    action='store_true',
                    help='指定使用默认远程配置参数连接Milvus数据库')
# parser.add_argument('-l', '--local', type=bool, default=True,
#                     action='store_ture',
#                     help='指定使用默认本地配置参数连接Milvus数据库。\n'
#                          '当与`--remote`选项同时指定时，`--remote`优先级高')
# 连接参数
parser.add_argument('-a', '--alias', type=str, default='',
                    help='指定参数连接Milvus数据库时的别名')
parser.add_argument('-p', '--port', type=str, default='',
                    help='指定参数连接Milvus数据库时的端口号')
parser.add_argument('--host', type=str, default='',
                    help='指定参数连接Milvus数据库时的主机地址')
parser.add_argument('-u', '--user', type=str, default='',
                    help='指定参数连接Milvus数据库时的用户名')
parser.add_argument('--passwd', type=str, default='',
                    help='指定参数连接Milvus数据库时的用户密码')
# 集合参数
parser.add_argument('--create_collect', type=bool, default=True,
                    help='指定Milvus数据库是否的集合')
parser.add_argument('-cn', '--cname', type=str, default='',
                    help='指定创建Milvus数据库的集合时的集合名称')
parser.add_argument('-es', '--embedding_size', type=int, default=512,
                    help='指定创建Milvus数据库的集合时的特征信息维度')
parser.add_argument('-d', '--dist_method', type=str, default='',
                    help='指定Milvus数据库用来计算特征向量之间距离的计算方式，默认是`L2`')

parser.add_argument('-nl', '--nlist', type=int, default=2048,
                    help='指定创建Milvus数据库的集合时的索引的`nlist`参数')

# ##################### Extract Feature ###################
# including model setting (only use clip4clip now)
parser.add_argument('-m', '--model', type=str, default='',
                    help='指定抽取特征的模型名称，这里只使用`clip4clip`模型')
parser.add_argument('-ef', '--extract_feature', type=bool, default=True,
                    help='指定是否对数据集抽取特征')
parser.add_argument('--hit_ratio',
                    action='store_true',
                    default=False,
                    help='计算模型的准确度信息，这需要数据集提供文本和视频对信息')

# ##################### Web Config ########################
# parser.add_argument('--run', type=bool, default=True,
#                     help='是否启用web服务')
parser.add_argument('--web_port', type=int, default=6650,
                    help='web服务启动端口')
parser.add_argument('--debug', default=False,
                    action='store_true',
                    help='是否启动web服务的debug模式')
parser.add_argument('--video_path', type=str, default=None,
                    help='视频数据集的文件目录名，默认值是`--path`参数指定的目录')
parser.add_argument('--gif_path', type=str, default=None,
                    help='指GIF数据的文件目录名，默认值是`--path`参数指定的目录下`gifs`目录')

# ##################### Run Config ########################






