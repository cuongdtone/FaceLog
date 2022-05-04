import os
import logging
from requests.api import head
from requests.sessions import Request
from .data_access import update_lc_employees_db
from datetime import datetime
import json
from pathlib import Path
from typing import List
import httpx
# import orjson
import ntpath
import requests
import sys
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii


def get_password(OOOO0O00O000OOOO0, OO0O0O00OO00OO000, OOOO00OOOOOO0OOOO):  # line:1
    OOOOO00O0000O000O = RSA.importKey(open(OO0O0O00OO00OO000, "r").read())  # line:2
    O0O00O0O000OO00O0 = RSA.importKey(open(OOOO0O00O000OOOO0, "r").read())  # line:3
    O00000OOO00OOOO00 = binascii.unhexlify(OOOO00OOOOOO0OOOO.encode('utf-8'))  # line:4
    O0OO000O0O000O000 = PKCS1_OAEP.new(O0O00O0O000OO00O0)  # line:5
    OOO0O0OOOO0OOOOO0 = O0OO000O0O000O000.decrypt(O00000OOO00OOOO00)  # line:6
    return OOO0O0OOOO0OOOOO0.decode()


def get_token(url: str, username: str, password: str):
    try:
        payload = {
            "username": username,
            "password": password
        }
        header = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        res = requests.post(url + "/auth/login", data=json.dumps(payload), headers=header)
        token = json.loads(res.text)["jwt_token"]
        return token
    except Exception as exc:
        _, _, exc_tb = sys.exc_info()
        print("Error in get_token at %d: %s" % (exc_tb.tb_lineno, str(exc)))
        logging.warning('core->get_token: %s' % str(exc))
    return ""


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def upload_multi_img(TOKEN: str, url: str, lc_files: List[str]) -> httpx.Response:
    """
    Send batch to FastAPI server.
    """
    header = {
        "Authorization": TOKEN
    }
    payload = {}
    files = [('files', (file, open(file, 'rb'), 'application/octet-stream')) for file in lc_files]
    res = requests.request("POST", url, data=payload, files=files, headers=header)

    if res.status_code == 200:
        return True
    return False


def post_timekeepings(URL_API: str, TOKEN: str, list_dict_timekeepings):
    try:
        # print("Syncing timekeeping...")
        if len(list_dict_timekeepings) > 0:
            list_image = [timekeeping["image"] for timekeeping in list_dict_timekeepings if
                          os.path.isfile(timekeeping["image"])]
            if len(list_image) > 0:
                upload_multi_img(TOKEN, URL_API + "/timekeeping/upload", list_image)

            prefix = "/timekeeping"
            payload = {
                "timekp_insert": list_dict_timekeepings
            }
            header = {
                'Content-Type': 'application/json',
                'accept': 'application/json',
                "Authorization": TOKEN
            }
            resp = requests.request("POST", url=URL_API + prefix, headers=header, data=json.dumps(payload))
            if resp.status_code == 200:
                code_checkin_list = [(x["code"], x["checkin"]) for x in list_dict_timekeepings]
                return code_checkin_list
    except Exception as exc:
        print("Error in post_timekeepings: %s" % str(exc))
        logging.warning('core->post_timekeepings: %s' % str(exc))
    return []


def post_employees(URL_API: str, TOKEN: str, DEVICE_ID: str, list_dict_employees):
    try:
        # print("Syncing data from local to server")
        if len(list_dict_employees) > 0:
            list_image = []
            # face_image_paths = list(Path('media/data').rglob('*.[jp][pn]*'))
            # user_codes = [x['code'] for x in list_dict_employees]
            # for i, path in enumerate(face_image_paths):
            #     code = str(str(path.name).split('_')[0])
            #     if code in user_codes:
            #         list_image.append(str(path.as_posix()))
            for obj_employee in list_dict_employees:
                if obj_employee['img_data'] is not None:
                    img_data = obj_employee['img_data']
                    arr_image = img_data.split('|')
                    list_image.extend(
                        [f'media/data/{name}' for name in arr_image if Path(f'media/data/{name}').is_file()])
                    # for img_path in arr_image:
                    #     # check exists file
                    #         # add imlist
                    #         # list_image.append(str(path.as_posix()))
                    obj_employee['img_data'] = arr_image

            # avatar_image_paths = list(Path('media/avatar').rglob('*.[jp][pn]*'))
            # for path in avatar_image_paths:
            #     code = str(str(path.name).split('.')[0])
            #     if code in user_codes:
            #         list_image.append(str(path.as_posix()))
            upload_multi_img(TOKEN, URL_API + "/timekeeping/upload", list_image)

            prefix = "/employee"
            payload = {
                "list_employee": list_dict_employees,
                "device_id": DEVICE_ID
            }
            header = {
                'Content-Type': 'application/json',
                'accept': 'application/json',
                "Authorization": TOKEN
            }

            resp = requests.request("POST", url=URL_API + prefix, headers=header, data=json.dumps(payload))
            if resp.status_code == 200:
                code_list = [x["code"] for x in list_dict_employees]
                return code_list
    except Exception as exc:
        print("Error in post_employees: %s" % str(exc))
        logging.warning('core->post_employees: %s' % str(exc))
    return []


def get_sv_list_employees(URL_API: str, TOKEN: str, DEVICE_ID: str, limit: int, skip: int = 0):
    try:
        prefix = "/employee?limit=%s&skip=%s&device_id=%s" % (limit, skip, DEVICE_ID)
        header = {
            "Authorization": TOKEN
        }
        resp = requests.request("GET", URL_API + prefix, headers=header)
        json_data = json.loads(resp.text)
        if resp.status_code == 200 and "data" in json_data:
            return json_data["data"]
    except Exception as exc:
        print("Error in get_sv_list_employees: %s" % str(exc))
        logging.warning('core->get_sv_list_employees: %s' % str(exc))
    return []


def sync_data_from_server_to_local(DB_PATH: str, list_lc_employees, list_sv_employees):
    try:
        # print("Syncing data server to local")
        for sv_employee in list_sv_employees:
            local_employee = [x for x in list_lc_employees if x['code'] == sv_employee['code']]
            if len(local_employee) > 0:
                # update
                # sv_date = datetime.strptime(sv_employee["updated_date"], "%Y-%m-%d %H:%M:%S")
                # if local_employee[0]['updated_date'] != sv_date:
                update_lc_employees_db(DB_PATH, True, sv_employee)
            else:
                # insert
                update_lc_employees_db(DB_PATH, False, sv_employee)
    except Exception as exc:
        _, _, exc_tb = sys.exc_info()
        print("Error in sync_data_from_server_to_local: %s" % str(exc))
        logging.warning('core->sync_data_from_server_to_local: %s' % str(exc))


def download_image_from_employee_code(URL_API: str, TOKEN: str, DATA_PATH: str, list_img: List[str]):
    try:
        img_data = ''
        for img in list_img:
            prefix = "/timekeeping/download"
            payload = {
                "image_path": "media/" + img
            }
            header = {
                'Content-Type': 'application/json',
                "Authorization": TOKEN
            }
            resp = requests.get(
                URL_API + prefix,
                data=json.dumps(payload),
                headers=header
            )
            if resp.status_code == 200:
                file_name = img.replace('\\', '/').split('/')[-1]
                img_path = os.path.join(DATA_PATH, file_name)
                Path(ntpath.split(img_path)[0]).mkdir(parents=True, exist_ok=True)
                open(img_path, "wb").write(resp.content)
                img_data += file_name + '|'
            else:
                print("Error in download %s" % img)
        if len(img_data) > 1:
            img_data = img_data[:-1]
        return img_data
    except Exception as exc:
        _, _, exc_tb = sys.exc_info()
        print("Error in download_image_from_employee_code: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
        logging.warning('core->download_image_from_employee_code: %s' % str(exc))
    return None


def download_image(URL_API: str, TOKEN: str, DATA_PATH: str, sv_file_name: str):
    try:
        prefix = "/timekeeping/download"
        payload = {
            "image_path": "media/" + sv_file_name
        }
        header = {
            'Content-Type': 'application/json',
            "Authorization": TOKEN
        }
        resp = requests.get(
            URL_API + prefix,
            data=json.dumps(payload),
            headers=header
        )
        if resp.status_code == 200:
            file_name = sv_file_name.replace('\\', '/').split('/')[-1]
            img_path = os.path.join(DATA_PATH, file_name)
            Path(ntpath.split(img_path)[0]).mkdir(parents=True, exist_ok=True)
            open(img_path, "wb").write(resp.content)
        else:
            print("Error in download %s" % sv_file_name)

        return True
    except Exception as exc:
        _, _, exc_tb = sys.exc_info()
        print("Error in download_image_from_employee_code: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
        logging.warning('core->download_image_from_employee_code: %s' % str(exc))
    return None


def delete_image_employee(LOCAL_PATH: str, employee_code: str, list_lc_employee):
    try:
        result = [x for x in list_lc_employee if x["code"] == employee_code]
        if result is not None and len(result) > 0:
            employee = result[0]
            if employee is not None and employee["img_data"] is not None:
                list_image = []
                img_data = employee['img_data']
                arr_image = img_data.split('|')
                list_image.extend(
                    [f'./media/data/{name}' for name in arr_image if Path(f'{LOCAL_PATH}{name}').is_file()])
                for img in list_image:
                    print(img)
                    os.unlink(img)
        return True
    except Exception as exc:
        _, _, exc_tb = sys.exc_info()
        print("Error in delete_image_employee: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
        logging.warning('core->delete_image_employee: %s' % str(exc))
    return None
