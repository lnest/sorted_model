# -*- coding: utf-8 -*-

# ------------------------------------
# Create On 2018/6/2 15:51 
# File Name: time_eval.py
# Edit Author: lnest
# ------------------------------------

import time
from functools import wraps


def time_count(func):
    s_time = time.time()

    @wraps(func)
    def wrapper(*args, **kwargs):
        data = func(*args, **kwargs)
        e_time = time.time()
        print('run <{}> cost: {}'.format(func.__name__, e_time - s_time))
        return data
    return wrapper


class TimeCountBlock:
    def __init__(self, name=None):
        self._name = name
        self._timepast = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, enter_type, value, traceback):
        self._end = time.time()
        self._timepast = self._end - self._start
        print(u'Runing block <{}> cost time: {}'.format(self._name, round(self._timepast, 3)))
