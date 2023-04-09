class BaseConfig:

    # 集合相关
    collection_name = 'text_video_retrieval'
    embedding_dim = 512
    dist_calcu_method = 'L2'
    nlist = 2048


class MilvusConfig(BaseConfig):

    def __init__(self,
                 alias='test',
                 host: str = 'localhost',
                 port: str = '19530',
                 username: str = None,
                 password: str = None):
        # 连接相关
        self.alias = alias
        self.host = host
        self.port = port
        self.username = username
        self.password = password


local_milvus_config = MilvusConfig()
remote_milvus_config = MilvusConfig(host='ssh.whq6.top', port='6612')

__all__ = [local_milvus_config, remote_milvus_config, MilvusConfig]


