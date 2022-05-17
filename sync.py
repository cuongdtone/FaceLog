import time

from utils.core import delete_image_employee, download_image_from_employee_code, get_password, get_sv_list_employees, \
    get_token, \
    post_employees, post_timekeepings, sync_data_from_server_to_local, download_image, delete_image_employee
from utils.data_access import get_employee_from_local, get_timekeeping_from_local, update_status_employees, \
    update_status_timekeeping
import yaml
import logging
from datetime import datetime
from pathlib import Path
import os
import sys

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(filename="./logs/" + datetime.now().strftime('%Y%m%d') + '.log',
                    filemode='a',
                    format='%(levelname)s:\t%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.ERROR)

if __name__ == "__main__":
    print("Load config...")
    yaml_file = open("src/settings.yaml", 'r', encoding='utf-8')
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    URL_API = str(cfg["URL_API"])
    DB_PATH = cfg["DB_PATH"]
    DEVICE_ID = cfg["device_name"]
    USERNAME = cfg["username_login"]
    PASSWORD = cfg["password_login"]
    PUBLIC_KEY = cfg['public_key_path']
    PRIVATE_KEY = cfg['private_key_path']
    DATA_PATH = cfg['data_path']
    AVATAR_PATH = cfg['avatar_path']
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    TOKEN = get_token(URL_API, USERNAME, get_password(PRIVATE_KEY, PUBLIC_KEY, PASSWORD))
    if TOKEN is None or TOKEN == "":
        time.sleep(60)

    while True:
        try:
            yaml_file = open("src/settings.yaml", 'r', encoding='utf-8')
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            DB_PATH = cfg["DB_PATH"]
            DEVICE_ID = cfg["device_name"]
            USERNAME = cfg["username_login"]
            PASSWORD = cfg["password_login"]
            PUBLIC_KEY = cfg['public_key_path']
            PRIVATE_KEY = cfg['private_key_path']
            DATA_PATH = cfg['data_path']
            AVATAR_PATH = cfg['avatar_path']

            if TOKEN is None or TOKEN == "" or URL_API != str(cfg["URL_API"]):
                URL_API = str(cfg["URL_API"])
                TOKEN = get_token(URL_API, USERNAME, get_password(PRIVATE_KEY, PUBLIC_KEY, PASSWORD))
                if TOKEN is None or TOKEN == "":
                    time.sleep(60)
                    continue
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            print("Error __main__: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
            logging.warning('sync-> sync employee: %s' % str(exc))

        try:
            # lay all danh sach employee db local: status=-1
            list_dict_employees = get_employee_from_local(DB_PATH)

            # get danh sách employee server:
            list_sv_employees = get_sv_list_employees(URL_API, TOKEN, DEVICE_ID, 100)
            # download anh ve
            for employee in list_sv_employees:
                delete_image_employee(DATA_PATH, employee['code'], list_dict_employees)
                img_data = download_image_from_employee_code(URL_API, TOKEN, os.path.join(CURRENT_PATH, DATA_PATH),
                                                             employee['img_data'])
                employee['data'] = img_data

                if employee['avatar'] is not None and str(employee['avatar']) != "":
                    download_image(URL_API, TOKEN, os.path.join(CURRENT_PATH, AVATAR_PATH), employee['avatar'])

            # dong bo tu server ve local
            sync_data_from_server_to_local(DB_PATH, list_dict_employees, list_sv_employees)

            # update device_id -> cho danh sach list_sv_employees: sync_device_ids + device_id
            # lay danh sach employee db local can dong bo: status = 0
            list_dict_employees = get_employee_from_local(DB_PATH, status=0)
            list_code = post_employees(URL_API, TOKEN, DEVICE_ID, list_dict_employees)
            # todo: chua cap nhat device_id -> sua api
            if len(list_code) > 0:
                update_status_employees(DB_PATH, list_code)  # cap nhat status = 1
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            print("Error __main__: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
            logging.warning('sync-> sync employee: %s' % str(exc))

        try:
            # sync timekeeping lên server
            # lay danh sach timekeeping db local
            list_dict_timekeepings = get_timekeeping_from_local(DB_PATH)
            list_code_checkin = post_timekeepings(URL_API, TOKEN, list_dict_timekeepings)
            if len(list_code_checkin) > 0:
                # update status đã sync
                update_status_timekeeping(DB_PATH, list_code_checkin)
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            print("Error __main__: %s - Line: %d" % (str(exc), exc_tb.tb_lineno))
            logging.warning('sync-> sync timekeeping: %s' % str(exc))
        time.sleep(5)
