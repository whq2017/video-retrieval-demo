from .result import Result
from .web import app
from flask import request, jsonify


@app.get('/api/search_by_text')
def search_by_text():
    search_key = request.args['key']
    # 获取全局对象
    search = app.config['search']
    dataset = app.config['dataset']

    id_list = search.search_videos(search_key=[search_key])
    print(id_list)
    result = dataset.get_info(id_list[search_key], search_key)

    return jsonify(Result.successWithData(result))


# @app.post('/api/search_by_texts')
# def search_by_texts():
#     search_key = json.loads(request.get_data(as_text=True))
#     print(search_key)
#     search = app.config['search']
#
#     result = search.search_videos(search_key=search_key)
#     print(result)
#     return jsonify(Result.successWithData(result))





