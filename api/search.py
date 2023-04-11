import json
from .result import Result
from .web import app
from flask import request, jsonify


@app.get('/api/search_by_text')
def search_by_text():
    search_key = request.args['key']
    print(search_key)
    search = app.config['search']
    if isinstance(search_key, str):
        search_key = [search_key]

    result = search.search_videos(search_key=search_key)
    print(result)
    return jsonify(Result.successWithData(result))


@app.post('/api/search_by_texts')
def search_by_texts():
    search_key = json.loads(request.get_data(as_text=True))
    print(search_key)
    search = app.config['search']

    result = search.search_videos(search_key=search_key)
    print(result)
    return jsonify(Result.successWithData(result))





