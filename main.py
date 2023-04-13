import os

from load_data import MSRVTT
from milvus_server import MilvusCollection
from config import local_milvus_config, remote_milvus_config
from extract_feature import VideoSearch


from commands import parser
from web_config import WebConfig
from api import web


def init_web_config(args, dataset, search):
    # ##################### Web Config ########################
    # run = args.run
    debug = args.debug
    web_config = WebConfig(dataset=dataset,
                           search=search,
                           # run=run,
                           debug=debug)
    if args.web_port > 1000:
        web_config.web_port = args.web_port
    # if args.video_path != '' and args.video_path is not None:
    #     web_config.video_path = args.video_path
    # if args.gif_path != '' and args.gif_path is not None:
    #     web_config.gif_path = args.gif_path
    return web_config


def do_extract_feature(args, dataset, milvus_config):
    # ##################### Extract Feature ###################
    # model = args.model
    extract_feature = args.extract_feature
    hit_ratio = args.hit_ratio
    # 创建VideoSearch处理视频和文本特征信息
    search = VideoSearch(dataset, milvus_config)
    if extract_feature:
        # 抽取特征，这个操作也只需要执行一次，抽取一次信息即可
        search.extract_feature_all_video()
    if hit_ratio:
        # 统计测试集所有的文本特征，计算模型准确率
        result = search.get_hit_ratio()
        print(f"Benchmark: {result}")
    return search


def init_milvus(args):
    # ##################### 创建Milvus配置文件 #####################
    milvus_config = remote_milvus_config() if args.remote else local_milvus_config()
    if args.alias != '' and args.alias is not None:
        milvus_config.alias = args.alias
    if args.port != '' and args.port is not None:
        milvus_config.port = args.port
    if args.host != '' and args.host is not None:
        milvus_config.host = args.host
    if args.user != '' and args.user is not None:
        milvus_config.user = args.user
    if args.passwd != '' and args.passwd is not None:
        milvus_config.password = args.passwd
    # ##################### 创建Milvus Collection配置文件 #####################
    create_collect = args.create_collect
    if create_collect:
        if args.cname != '' and args.cname is not None:
            milvus_config.collection_name = args.cname
        if args.embedding_size > 0:
            milvus_config.embedding_dim = args.embedding_size
        if args.dist_method != '' and args.dist_method is not None:
            milvus_config.dist_calcu_method = args.dist_method
        if args.nlist > 0:
            milvus_config.nlist = args.nlist
        # 创建MilvusCollection对象
        # 这个操作只需要执行一次，用来创建collection集合就可以了
        with MilvusCollection.new(milvus_config) as milvus:
            # 创建Milvus数据库Collection
            milvus.create_milvus_collection()
    return milvus_config


def init_dataset(args):
    # ##################### 处理数据集数据集 ####################
    create_gif = args.create_gif
    num_samples = 16 if args.num < 0 else args.num
    path = 'datasets' if args.path == '' or args.path is None else args.path
    # create dataset
    dataset = MSRVTT(path=path, create_gif=create_gif, num_samples=num_samples)
    return dataset


def pipe(args):
    # 处理数据集数据集
    dataset = init_dataset(args)
    # 制作gif动图
    dataset.recreate_gifs()

    # 创建配置文件
    config = local_milvus_config()

    # 创建MilvusCollection对象
    # 这个操作只需要执行一次，用来创建collection集合就可以了
    with MilvusCollection.new(config) as milvus:
        # 创建Milvus数据库Collection
        milvus.create_milvus_collection()

    # 创建VideoSearch处理视频和文本特征信息
    search = VideoSearch(dataset, config)

    # 抽取特征，这个操作也只需要执行一次，抽取一次信息即可
    search.extract_feature_all_video()
    # 统计测试集所有的文本特征，计算模型准确率
    result = search.get_hit_ratio()
    print(f"Benchmark: {result}")

    # 通过一个文本获取相关视频信息
    search_key = ['baseball player', 'looking at data']
    result = search.search_videos(search_key=search_key)
    print(f'Search {search_key} result:\n{result}')


def init_system_config(args):

    dataset = init_dataset(args)

    milvus_config = init_milvus(args)

    search = do_extract_feature(args, dataset, milvus_config)

    web_config = init_web_config(args, dataset, search)

    return dataset, search, web_config


if __name__ == '__main__':
    _args = parser.parse_args()
    if _args.run == 'pipe':
        pipe(_args)
        exit(0)
    # 解析参数
    _dataset, _search, _web_config = init_system_config(_args)
    # 启动web服务
    web.run(_web_config)

