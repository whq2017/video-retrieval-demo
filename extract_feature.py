import os

import pandas as pd
from towhee.dc2 import ops, pipe, DataCollection
from config import MilvusConfig, local_milvus_config
from load_data import MSRVTT


def get_device():
    # 检查cuda命令
    cmd = 'which nvcc > /dev/null'
    result = os.system(cmd)
    print(f"CUDA is {'not ' if result != 0 else ''}exist.")
    return 'cuda:0' if result == 0 else 'cpu'


class VideoSearch:

    def __init__(self, dataset: MSRVTT,
                 config: MilvusConfig = local_milvus_config,
                 device: str = get_device()):
        self.device = device
        self._dataset = dataset
        self._config = config

    def search_videos(self, search_key: list[str], show_num=10) -> list[dict[str]:list]:
        search_func = (
            pipe.input('sentence_list')
            .map('sentence_list', 'sentence', lambda x: x)
            .map('sentence', 'vec',
                 ops.video_text_embedding.clip4clip(model_name='clip_vit_b32',
                                                    modality='text',
                                                    device=self.device))
            .map('vec', 'rows',
                 ops.ann_search.milvus_client(host=self._config.host,
                                              port=self._config.port,
                                              collection_name=self._config.collection_name,
                                              limit=show_num))
            .map('rows', 'videos_path',
                 lambda rows: (os.path.join(self._dataset.video_path,
                                            'video' + str(r[0]) + '.mp4') for r in rows))
            .output('videos_path')
        )

        return [{key: search_func(key).get()[0]} for key in search_key]

    def extract_feature_all_video(self, num_samples: int = 12):
        def range_pd(data: pd.DataFrame):
            for row in data.itertuples(index=False):
                yield getattr(row, 'id'), getattr(row, 'video_path')

        extract_func = (
            pipe.input('csv_content')
            .flat_map('csv_content', ('id', 'video_path'), range_pd)
            .map('video_path', 'frames',
                 ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample',
                                         args={'num_samples': num_samples}))
            .map('frames', 'vec',
                 ops.video_text_embedding.clip4clip(model_name='clip_vit_b32',
                                                    modality='video',
                                                    device=self.device))
            .map(('id', 'vec'), (),
                 ops.ann_insert.milvus_client(host=self._config.host,
                                              port=self._config.port,
                                              collection_name=self._config.collection_name))
            .output()
        )
        extract_func(self._dataset.csv_file_content.loc[:, ['id', 'video_path']])

    def get_hit_ratio(self, show_num: int = 10) -> pd.DataFrame:
        def mean_hit_ratio(actual, *predicteds):
            rets = []
            for predicted in predicteds:
                ratios = []
                for act, pre in zip(actual, predicted):
                    hit_num = len(set(act) & set(pre))
                    ratios.append(hit_num / len(act))
                rets.append(sum(ratios) / len(ratios))
            return rets

        def get_label_from_raw_data(data):
            return [item[0] for item in data]

        def range_pd(data: pd.DataFrame):
            for row in data.itertuples(index=False):
                yield getattr(row, 'id'), getattr(row, 'sentence')

        dc_search = (
            pipe.input('csv_content')
            .flat_map('csv_content', ('id', 'sentence'), range_pd)
            .map('sentence', 'vec',
                 ops.video_text_embedding.clip4clip(model_name='clip_vit_b32',
                                                    modality='text',
                                                    device=self.device))
            .map('vec', 'top10_raw_res',
                 ops.ann_search.milvus_client(host=self._config.host,
                                              port=self._config.port,
                                              collection_name=self._config.collection_name,
                                              limit=show_num))
            .map('top10_raw_res', ('top1', 'top5', 'top10'), lambda x: (x[:1], x[:5], x[:10]))
            .map('id', 'ground_truth', lambda x: x)
            .output('id', 'sentence', 'ground_truth', 'top1', 'top5', 'top10')
        )

        ev = (
            pipe.input('dc_data')
            #
            .flat_map('dc_data', 'data', lambda x: x)
            .map('data', ('ground_truth', 'top1', 'top5', 'top10'),
                 lambda x: ([x['ground_truth']],
                            get_label_from_raw_data(x.top1),
                            get_label_from_raw_data(x.top5),
                            get_label_from_raw_data(x.top10))
                 )
            .window_all(('ground_truth', 'top1', 'top5', 'top10'),
                        ('top1_mean_hit_ratio', 'top5_mean_hit_ratio', 'top10_mean_hit_ratio'), mean_hit_ratio)
            .output('top1_mean_hit_ratio', 'top5_mean_hit_ratio', 'top10_mean_hit_ratio')
        )

        ret = dc_search(self._dataset.csv_file_content.loc[:, ['id', 'sentence']])
        benchmark = ev(DataCollection(ret))

        return benchmark.get()
