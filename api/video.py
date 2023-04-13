from flask import Response, jsonify
from .web import app
from .result import Result


@app.get('/api/video/<videoId>')
def video_content(videoId):
    def generate(path, size=1024):
        with open(path, 'rb') as video:
            data = video.read(size)
            while data:
                yield data
                data = video.read(size)

    dataset = app.config['dataset']

    file_path = dataset.get_video_fp(videoId)
    if file_path is not None:
        return Response(generate(file_path, size=2048), mimetype='video')
    return jsonify(Result.fail('not video'))

