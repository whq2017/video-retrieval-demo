import os

import pandas as pd
import towhee
from towhee.dc2 import ops, pipe
from pymilvus import Collection
from config import MilvusConfig, local_milvus_config
from load_data import MSRVTT


def get_device():
    # 检查cuda命令
    cmd = 'which nvcc > /dev/null'
    result = os.system(cmd)
    print(f"CUDA is exist ? {result == 0}")
    return 'cuda:0' if result == 0 else 'cpu'


class VideoSearch:

    def __init__(self, dataset: MSRVTT, collection: Collection, config: MilvusConfig = local_milvus_config, device: str = get_device()):
        self.device = device
        self._dataset = dataset
        self._collection = collection
        self._config = config

    def search_videos(self, search_key: list[str], show_num=10) -> list[pd.DataFrame]:
        search_func = (
            pipe.input('sentence_list')
            .map('sentence_list', 'sentence', lambda x: x)
            .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device=self.device))
            .map('vec', 'rows', ops.ann_search.milvus_client(host=self._config.host, port=self._config.port, collection_name=self._config.collection_name, limit=show_num))
            .map('rows', 'videos_path', lambda rows: (os.path.join(self._dataset.video_path, 'video' + str(r[0]) + '.mp4') for r in rows))
            .output('videos_path')
        )

        return [search_func(key).get()[0] for key in search_key]

    def extract_feature_all_video(self, num_samples: int = 12, store_batch: int = 30):
        (
            towhee.from_df(self._dataset.csv_file_content).unstream()
            .video_decode.ffmpeg['video_path', 'frames'](sample_type='uniform_temporal_subsample',
                                                         args={'num_samples': num_samples})
            .runas_op['frames', 'frames'](func=lambda x: [y for y in x])
            .video_text_embedding.clip4clip['frames', 'vec'](model_name='clip_vit_b32',
                                                             modality='video', device=self.device)
            .to_milvus['id', 'vec'](collection=self._collection, batch=store_batch)
        )

    def get_hit_ratio(self, show_num: int = 10) -> pd.DataFrame:
        dc = (
            towhee.from_df(self._dataset.csv_file_content).unstream()
            .video_text_embedding.clip4clip['sentence', 'text_vec'](model_name='clip_vit_b32',
                                                                    modality='text', device=self.device)
            .milvus_search['text_vec', 'top10_raw_res'](collection=self._collection, limit=show_num)
            .runas_op['video_id', 'ground_truth'](func=lambda x: [int(x[-4:])])
            .runas_op['top10_raw_res', 'top1'](func=lambda res: [x.id for i, x in enumerate(res) if i < 1])
            .runas_op['top10_raw_res', 'top5'](func=lambda res: [x.id for i, x in enumerate(res) if i < 5])
            .runas_op['top10_raw_res', 'top10'](func=lambda res: [x.id for i, x in enumerate(res) if i < 10])
        )
        benchmark = (
            dc.with_metrics(['mean_hit_ratio', ])
            .evaluate['ground_truth', 'top1'](name='recall_at_1')
            .evaluate['ground_truth', 'top5'](name='recall_at_5')
            .evaluate['ground_truth', 'top10'](name='recall_at_10')
            .report()
        )

        return benchmark
