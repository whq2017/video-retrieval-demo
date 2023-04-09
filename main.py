from load_data import MSRVTT
from milvus_server import MilvusCollection
from config import local_milvus_config, remote_milvus_config
from extract_feature import VideoSearch


if __name__ == '__main__':

    # 处理数据集数据集
    dataset = MSRVTT()
    # 制作gif动图
    # dataset.recreate_gifs()

    # 创建配置文件
    config = local_milvus_config

    # 创建MilvusCollection对象
    with MilvusCollection.new(config) as milvus:
        # 添加数据到Milvus数据库中
        collection = milvus.create_milvus_collection()

        # 创建VideoSearch处理视频和文本特征信息
        search = VideoSearch(dataset, collection, config)

        # 抽取特征
        search.extract_feature_all_video()
        # 统计测试集所有的文本特征，计算模型准确率
        # result = search.get_hit_ratio()

        # 通过一个文本获取相关视频信息
        search_key = ['baseball player', 'looking at data']
        result = search.search_videos(search_key=search_key)
        print(f'Search {search_key} result:\n{result}')


