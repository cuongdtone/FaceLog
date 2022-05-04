""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import datetime

timekeep = {}


def check_timekeep(side='front', user_code=''):
    key = f'{side}_{user_code}'
    now = datetime.datetime.now()
    # if len(timekeep) > 0:
    #     max_timekeep = max(timekeep.values())
    #     if (now - max_timekeep).total_seconds() < 10:
    #         return False
    if key not in timekeep.keys():
        timekeep.setdefault(key, now)
        return True
    if (now - timekeep[key]).total_seconds() > 3:
        timekeep[key] = now
        return True
    return False
