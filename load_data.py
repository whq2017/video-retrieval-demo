import os
import sys
import base64
import shutil
import zipfile
import requests
from io import BytesIO
import pandas as pd
from tqdm import tqdm

from IPython import display
from pathlib import Path
import towhee
from PIL import Image


def support_filename_encode(zip_file: zipfile.ZipFile, encoding: str = 'utf-8') -> zipfile.ZipFile:
    """
    zipfile读取zip包中的中文会使用`cp437`方式，导致出现乱码
    :param zip_file: 读取文件得到的ZipFile对象
    :param encoding: 希望改变后的编码方式，默认是 'utf-8'
    :return: 修改NameToInfo结构后的ZipFile对象
    """
    name_to_info = zip_file.NameToInfo
    # 先copy一个对象
    for name, info in name_to_info.copy().items():
        real_name = name.encode('cp437').decode(encoding)
        if real_name != name:
            # 修改文件名
            info.filename = real_name
            # 删除，并且重新设置文明名到文件的映射关系
            del name_to_info[name]
            name_to_info[real_name] = info

    return zip_file


def download_dataset(file_path: str, file_url: str, chunk_size: int = 1024) -> None:
    """
    通过网络下载所需要的数据集攻击
    :param file_path: 文件保存路径
    :param file_url: 数据集下载url
    :param chunk_size: 分块下载时分块大小
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    req = requests.get(file_url, stream=True)
    file_size = int(req.headers.get('Content-Length')) / chunk_size
    with open(file_path, 'wb') as fp:
        for item in tqdm(iterable=req.iter_content(chunk_size),
                         desc='download dataset',
                         total=file_size):
            fp.write(item)

    print(f'Download success! Dataset path is: {file_path}')


class MSRVTT:

    # 获取当前工作目录
    __WORK_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

    # 数据集下载相关信息
    __URL: str = 'https://github.com/towhee-io/examples/releases/download/data/text_video_search.zip'
    __DIR_NAME: str = 'text_video_search'
    __DATASET_NAME: str = 'text_video_search.zip'
    __GIF_DIR_NAME: str = 'gifs'

    # 数据集文件名
    __COMPRESS_VID_DIR: str = 'test_1k_compress'
    __VID_TEST_DESC_INFO_FILE_NAME: str = 'MSRVTT_JSFUSION_test.csv'

    # 属性信息
    __DATASET_CSV_CONTENT: pd.DataFrame

    def __init__(self, path: str = 'datasets', download: bool = True, chunk_size: int = 1024 * 10,
                 create_gif: bool = False, num_samples: int = 16):
        # 下载数据集相关
        self._path = path
        self._download_file_path = os.path.join(self._path, self.__DATASET_NAME)
        self._download = download
        self._chunk_size = chunk_size
        self._num_samples = num_samples

        # 读取文件信息
        self._root_dir = os.path.join(self._path, self.__DIR_NAME)
        self._vid_dir = os.path.join(self._root_dir, self.__COMPRESS_VID_DIR)
        self._csv_file_name = os.path.join(self._root_dir, self.__VID_TEST_DESC_INFO_FILE_NAME)
        self._gifs_dir = os.path.join(self._root_dir, self.__GIF_DIR_NAME)

        # 开始下载数据集
        self._dataset = self._download_dataset()
        # 解压数据集
        self._compress_dataset()
        # 读取数据信息
        self._load_csv_info()
        if create_gif:
            self._create_gifs_from_all_video()

    def _download_dataset(self) -> zipfile.ZipFile:
        """
        下载数据集
        :return: 处理好文件名格式后的数据集zipfile.ZipFile对象
        """
        if not os.path.exists(self._download_file_path):
            if self._download:
                Path(self._path).mkdir(exist_ok=True)
                download_dataset(self._download_file_path, self.__URL, self._chunk_size)
            else:
                print("Error: Dataset is not exist.Please use arg `download=True` to start download.")
                return sys.exit(1)
        return support_filename_encode(zipfile.ZipFile(self._download_file_path))

    def _compress_dataset(self) -> None:
        """
        解压文件信息到`self._path/self.__DIR_NAME`目录下
        """
        self._dataset.extractall(os.path.join(self._path, self.__DIR_NAME))

    def _load_csv_info(self):
        if not os.path.exists(self._csv_file_name):
            print(f"Error: Dataset file: {self._csv_file_name} is not exist. Please download dataset first.")
            sys.exit(1)

        self.__DATASET_CSV_CONTENT = pd.read_csv(self._csv_file_name)
        self.__DATASET_CSV_CONTENT['video_path'] = self.__DATASET_CSV_CONTENT.apply(
            lambda x: os.path.join(self._vid_dir, x['video_id']) + '.mp4', axis=1)
        self.__DATASET_CSV_CONTENT['id'] = self.__DATASET_CSV_CONTENT['video_id'].apply(lambda x: int(x[-4:]))

    def _create_gifs_from_all_video(self):
        self.recreate_gifs(show_vid_num=0)

    def recreate_gifs(self, show_vid_num=10) -> display.HTML:
        """
        删除所有gifs信息，重新为所有视频创建gif文件
        :param show_vid_num: 需要展示的gif文件个数，默认是10，方便测试使用
        :return: 添加需要展示gif后的display.HTML对象
        """
        path = Path(self._gifs_dir)
        # 如果创建重新创建，这删除所有之前的信息
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)

        video_path_list = self.__DATASET_CSV_CONTENT['video_path'].to_list()
        text_list = self.__DATASET_CSV_CONTENT['sentence'].to_list()

        gif_path_list = []
        for video_path in tqdm(video_path_list, desc='creating gif'):
            video_name = str(Path(video_path).name).split('.')[0]
            # 这里Path继承了PurePath，PurePath中重载了除法运算符
            gif_path = path / (video_name + '.gif')
            self.__convert_video2gif(video_path, gif_path, self._num_samples)
            gif_path_list.append(gif_path)
        # 这里返回的展示信息只有前show_vid_num个
        return self.__display_gif(gif_path_list[:show_vid_num], text_list[:show_vid_num])

    @staticmethod
    def __display_gif(video_path_list: list[Path], text_list: list[Path]) -> display.HTML | None:
        if len(video_path_list) == 0 or len(text_list) == 0:
            return None
        html = ''
        for video_path, text in zip(video_path_list, text_list):
            html_line = '<img src="{}"> {} <br/>'.format(video_path, text)
            html += html_line
        return display.HTML(html)

    @staticmethod
    def __convert_video2gif(video_path, output_gif_path, num_samples=16):
        frames = (
            towhee.glob(video_path)
            .video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': num_samples})
            .to_list()[0]
        )
        imgs = [Image.fromarray(frame) for frame in frames]
        imgs[0].save(fp=output_gif_path, format='GIF', append_images=imgs[1:], save_all=True, loop=0)

    @property
    def csv_file_content(self) -> pd.DataFrame:
        return self.__DATASET_CSV_CONTENT.copy()

    @property
    def video_path(self) -> str:
        return self._vid_dir

    def get_info(self, ids: list, searchKey: str):
        def get_gif_data(vid_path, fmt='gif'):
            output_buffer = BytesIO()
            self.__convert_video2gif(vid_path, output_buffer, num_samples=5)
            bytes_val = output_buffer.getvalue()

            data_str = base64.b64encode(bytes_val).decode('utf-8')
            # print(data_str)
            return f'data:image/{fmt};base64,' + data_str

        ids = [int(x) for x in ids]
        result_df = self.__DATASET_CSV_CONTENT[self.__DATASET_CSV_CONTENT.id.isin(ids)]
        # print(result_df)

        return [{
                'videoId': str(row['id']),
                'sentence': row['sentence'],
                'searchKey': searchKey,
                'img': get_gif_data(row['video_path'])} for idx, row in result_df.iterrows()]

    def get_video_fp(self, video_id):
        video_path = os.path.join(self.__WORK_DIR_PATH, self._vid_dir, "video{}.mp4".format(video_id))
        print("video path: ", video_path)
        return video_path if Path(video_path).exists() else None

