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
        if user["embed"] is not None and len(user["embed"]) != 0:
            info = [user['fullname'], user['code'], str2np(user['embed'])]
            data.append(info)
    data = np.array(data)
    return data, users_info


def str2np(face_feature):
    str = face_feature.strip('[')
    str = str.strip(']')
    return np.array([float(i.strip('\n')) for i in str.split()])
