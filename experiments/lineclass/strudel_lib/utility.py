# Created by lan at 2021/11/9
from math import sqrt

import pandas

date_patterns = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%Y/%m/%d', '%y/%m/%d', '%d/%m/%y']
import datetime


def isDate(value):
    _isDate = False
    for date_pattern in date_patterns:
        try:
           datetime.datetime.strptime(value, date_pattern)
           _isDate = True
           break
        except:
            pass
    return _isDate


def detect_datatype(value):
    """
    :param value: the value whose data type is to be checked.
    :return: 0 - 'date', 1 - 'float', 2 - 'int', 3 - 'empty', 4 - 'string', 5 - 'null'
    """
    if value is None:
        return 5
    if isDate(value):
        return 0
    try:
        float(value)
        return 1 if '.' in value else 2
    except:
        if len(value) == 0:
            return 3
        else:
            return 4


def calculate_hist_diff(hist1, hist2):
    total_sum = sum([sqrt(e1 * e2) for e1, e2 in zip(hist1, hist2)])

    factor = sqrt(sum(hist1) * sum(hist2))
    if factor != 0.0:
        bhattacharyya_factor = 1 / factor

        hd = 1 - bhattacharyya_factor * total_sum
        if hd > 0.0:
            hd = sqrt(hd)
        else:
            hd = 0.0
    else:
        hd = 0.0
    return hd