""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

from .sqlite_database import get_all_employee
import os
import pickle

root_path = 'src/data/'
def load_user_data():
    users_info = get_all_employee(None)
    data = []
    for user in users_info:
        if user["pkl_file"] and str(user["pkl_file"]) != "" and os.path.exists(root_path + user["pkl_file"]):
            with open(root_path + user["pkl_file"], 'rb') as f:
                info = {'feet': pickle.load(f)}
                info.update({'fullname': user['fullname']})
                info.update({'code': user['code']})
                data.append(info)
    return data, users_info
