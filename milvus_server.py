from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from config import MilvusConfig


class MilvusCollection:

    def __init__(self, config: MilvusConfig):
        self._config = config

    def connect(self):
        print(f'alias: [{self._config.alias}] start to connect milvus server...')
        connections.connect(alias=self._config.alias,
                            host=self._config.host,
                            port=self._config.port,
                            user=self._config.username,
                            password=self._config.password)

        print(f'alias: [{self._config.alias}] connect milvus server successfully!')
        return connections.get_connection_addr(self._config.alias)

    def disconnect(self):
        print(f'alias: [{self._config.alias}] start to disconnect milvus server...')
        connections.disconnect(self._config.alias)
        print(f'alias: [{self._config.alias}] disconnect milvus server successfully!')

    def create_milvus_collection(self) -> Collection:

        if utility.has_collection(self._config.collection_name, using=self._config.alias):
            utility.drop_collection(self._config.collection_name, using=self._config.alias)

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors',
                        dim=self._config.embedding_dim)
        ]
        schema = CollectionSchema(fields=fields, description='video retrieval')
        collection = Collection(name=self._config.collection_name, schema=schema, using=self._config.alias)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type': self._config.dist_calcu_method,
            'index_type': "IVF_FLAT",
            'params': {"nlist": self._config.nlist}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("create collection successfully!")
        return collection

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    @staticmethod
    def new(config):
        return MilvusCollection(config)

    @property
    def connection(self):
        return connections.get_connection_addr(self._config.alias)

