
# ver1: db with csv
import pandas as pd
from datetime import datetime, timedelta
import os

class Log():
    def __init__(self, db='log'):

        self.db = db
    def login(self, person):
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        path_csv_today = self.db + '/' + today + '.csv'
        if not os.path.exists(path_csv_today):
            log = pd.DataFrame({'Name': [], 'Position': [], 'Time':[]})
            log.to_csv(path_csv_today, index=False)
        else:
            log = pd.read_csv(path_csv_today)


        if not person['Name'] in log.iloc[:, 0].tolist() and person['Name'] != 'uknown':
            attendance_info = pd.DataFrame({"Name": [person['Name']],
                                            "Position": [person['Position']],
                                            'Time': now.strftime('%a %H:%M:%S')
                                            })
            log = log.append(attendance_info)
            log.to_csv(path_csv_today, index=False)
            return True
        return False