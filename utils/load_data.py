""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

from .sqlite_database import get_all_employee
import os
import numpy as np
from annoy import AnnoyIndex


root_path = 'src/data/'

def load_user_data():
    users_info = get_all_employee(None)
    tree = AnnoyIndex(512, 'euclidean')
    data = []
    count = 0
    for user in users_info:
        if user["embed"] is not None and len(user["embed"]) != 0:
            info = [user['code'], user['fullname'], user['name']]
            feat = str2list(user['embed'])
            tree.add_item(count, feat)
            data.append(info)
            count +=1
    tree.build(100)
    data = np.array(data)
    return data, tree, users_info


def str2list(face_feature):
    str = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in str.split()]
