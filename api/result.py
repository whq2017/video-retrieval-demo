from typing import Any


class Result:
    @staticmethod
    def success():
        return {
            'code': 20000,
            'message': 'success',
            'data': None
        }

    @staticmethod
    def fail(message: str):
        return {
            'code': 10000,
            'message': message,
            'data': None
        }

    @staticmethod
    def successWithData(data: Any, message: str = 'success'):
        return {
            'code': 20000,
            'message': message,
            'data': data
        }
