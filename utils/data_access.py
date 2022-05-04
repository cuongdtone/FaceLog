from datetime import datetime
import ntpath
import sqlite3
import logging


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def modified_employed(dict_employee):
    if "birthday" not in dict_employee or dict_employee["birthday"] == "" or dict_employee["birthday"] is None:
        dict_employee["birthday"] = None
    # else:
    #     dict_employee["birthday"] = datetime.strptime(dict_employee["birthday"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

    if "last_timekeeping" not in dict_employee or dict_employee["last_timekeeping"] == "" or dict_employee[
        "last_timekeeping"] is None:
        dict_employee["last_timekeeping"] = None

    if "created_date" not in dict_employee or dict_employee["created_date"] == "" or dict_employee[
        "created_date"] is None:
        dict_employee["created_date"] = None

    if "updated_date" not in dict_employee or dict_employee["updated_date"] == "" or dict_employee[
        "updated_date"] is None:
        dict_employee["updated_date"] = None
    return dict_employee


def modified_timekeepings(dict_timekeeping):
    if "working_date" not in dict_timekeeping or dict_timekeeping["working_date"] == "" or dict_timekeeping[
        "working_date"] is None:
        dict_timekeeping["working_date"] = None
    else:
        dict_timekeeping["working_date"] = datetime.strptime(dict_timekeeping["working_date"], "%Y-%m-%d").strftime(
            "%Y-%m-%d %H:%M:%S")

    if "checkin" not in dict_timekeeping or dict_timekeeping["checkin"] == "" or dict_timekeeping["checkin"] is None:
        dict_timekeeping["checkin"] = None

    dict_timekeeping["image"] = str(dict_timekeeping["image"])
    return dict_timekeeping


def modified_datetime(dt: str):
    return dt.replace("Z", "").replace(",", "")


def get_namefile(path):
    return ntpath.split(path)[-1]


def update_lc_employees_db(DB_PATH, is_updated, dict_employee):
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = dict_factory
        cur = con.cursor()

        img_data = dict_employee["data"]

        if is_updated:
            query_update = "UPDATE employee set fullname = ?, name = ?, birthday = ?, position = ?, branch = ?, department = ?, avatar = ?, img_data = ?, vocative = ?, isadmin = ?, active = ?, updated_date = ?, updated_user = ? where code = ?"
            cur.execute(query_update, (
                dict_employee["fullname"],
                dict_employee["name"],
                dict_employee["birthday"],
                dict_employee["position"],
                dict_employee["branch"],
                dict_employee["department"],
                dict_employee["avatar"],
                dict_employee["data"],
                dict_employee["vocative"],
                dict_employee["isadmin"],
                dict_employee["active"],
                dict_employee["updated_date"],
                dict_employee["updated_user"],
                dict_employee["code"]
            ))
        else:
            query_update = "INSERT OR IGNORE INTO employee (code, fullname, name, birthday, position, branch, department , avatar, img_data, vocative, isadmin, active, created_date, created_user, updated_date, updated_user, status) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1);"
            cur.execute(query_update, (
                dict_employee["code"],
                dict_employee["fullname"],
                dict_employee["name"],
                dict_employee["birthday"],
                dict_employee["position"],
                dict_employee["branch"],
                dict_employee["department"],
                dict_employee["avatar"],
                dict_employee["data"],
                dict_employee["vocative"],
                dict_employee["isadmin"],
                dict_employee["active"],
                dict_employee["created_date"],
                dict_employee["created_user"],
                dict_employee["updated_date"],
                dict_employee["updated_user"]
            ))

        con.commit()
        con.close()
    except Exception as exc:
        print("Error in update_lc_employees_db: %s" % str(exc))
        logging.warning('data_access->update_lc_employees_db: %s' % str(exc))


def update_status_timekeeping(DB_PATH, code_checkin_list):
    try:
        query_update = "update timekeepings set status = ? where code = ? and checkin = ?"
        con = sqlite3.connect(DB_PATH)
        con.row_factory = dict_factory
        cur = con.cursor()
        for code, checkin in code_checkin_list:
            checkin = datetime.strptime(checkin, "%Y-%m-%d %H:%M:%S")
            cur.execute(query_update, (1, code, checkin))
            con.commit()
        con.close()
        return True
    except Exception as exc:
        print("Error in update_status_timekeeping: %s" % str(exc))
        logging.warning('data_access->update_status_timekeeping: %s' % str(exc))
    return False


def update_status_employees(DB_PATH, list_code: list):
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = dict_factory
        cur = con.cursor()
        for code in list_code:
            query_update = "update employee set status = 1 where code = '%s'" % code
            cur.execute(query_update)
            con.commit()
        cur.close()
        return True
    except Exception as exc:
        print("Error in update_status_employees: %s" % str(exc))
        logging.warning('data_access->update_status_employees: %s' % str(exc))
    return False


def get_timekeeping_from_local(DB_PATH):
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = dict_factory
        cur = con.cursor()
        list_dict_timekeepings = []
        for row in cur.execute(
                'select code, fullname, working_date, checkin, device_name, source, image from timekeepings where status = 0 limit 100'):
            list_dict_timekeepings.append(modified_timekeepings(row))
        con.close()
        return list_dict_timekeepings
    except Exception as exc:
        print("Error in get_timekeeping_from_local: %s" % str(exc))
        logging.warning('data_access->get_timekeeping_from_local: %s' % str(exc))
    return []


def get_employee_from_local(DB_PATH, status: int = -1):
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = dict_factory
        cur = con.cursor()
        list_dict_employees = []
        script = 'select * from employee '
        if status > -1:
            script = script + ' where status = ' + str(status)
        for row in cur.execute(script):
            list_dict_employees.append(modified_employed(row))
        con.close()
        return list_dict_employees
    except Exception as exc:
        print("Error in get_employee_from_local: %s" % str(exc))
        logging.warning('data_access->get_employee_from_local: %s' % str(exc))
    return []
