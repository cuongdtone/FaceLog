""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import datetime
import cv2
import numpy as np
import os
from threading import Thread
from .sqlite_database import insert_timekeeping
from datetime import datetime
import base64
from unidecode import unidecode

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


def checkin(people, checkin_list, employees, frame_ori, config):
    checkin_list_code = checkin_list.keys()
    now = datetime.now()
    timenow = now.strftime('%Y-%m-%d %H:%M:%S')
    if now.strftime('%H:%M:%S') == '00:00:00':
        return {}
    for info in people:
        code = info['code']
        if code is None:  # uknown person
            continue
        employee = [employee for employee in employees if employee['code'] == code][0]
        if not employee['active']:  # Not active in database
            continue
        # finished 2 case close door when face exist
        # Just checkin and return camera: checkin again
        if not code in checkin_list_code or (now - checkin_list[f'{code}']).total_seconds() > 0.7:
            print('Success') # opendoor and checkin code here
            text = unidecode(f'{code} - {employee["fullname"]} - front - {timenow}')
            cv2.putText(frame_ori, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            retval, buffer = cv2.imencode('.jpg', frame_ori)
            jpg_as_text = base64.b64encode(buffer)
            insert_timekeeping(None, code, employee["fullname"], config['device_name'], image=jpg_as_text)
        if not code in checkin_list_code:
            checkin_list.update({f'{code}': now})  # new track
        else:
            checkin_list[f'{code}'] = now  # update old track
    return checkin_list
