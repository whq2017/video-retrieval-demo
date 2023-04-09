from . import app
from flask import request


@app.get('/search_by_text')
def search_by_text():
    search_key = request.form['key']
    print(search_key)





