class WebConfig:

    def __init__(self, dataset, search,
                 # run: bool = True,
                 port: int = 6650,
                 debug: bool = False,
                 # video_path: str = None,
                 # gif_path: str = None
                 ):
        # self.run = run
        self.debug = debug
        self.port = port
        # self.video_path = video_path
        # self.gif_path = gif_path

        self.dataset = dataset
        self.search = search


