""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

from .sqlite_database import get_all_employee
import os
import numpy as np

root_path = 'src/data/'

def load_user_data():
    users_info = get_all_employee(None)
    data = []
    for user in users_info:
        if user["face_feature"] is not None and len(user["face_feature"]) != 0:
            info = {'feet': str2np(user["face_feature"])}
            info.update({'fullname': user['fullname']})
            info.update({'code': user['code']})
            data.append(info)
    return data, users_info


def str2np(face_feature):
    str = face_feature.strip('[')
    str = str.strip(']')
    float_lst = [float(i.strip('\n')) for i in str.split()]
    return np.array(float_lst, dtype=np.float)
