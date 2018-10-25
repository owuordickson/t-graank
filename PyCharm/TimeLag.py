"""
@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""

from datetime import datetime
import time


def get_time_diffs(dataset,time_type,step):

    time_diffs = []

    for i in range(1,len(dataset)):
        if i < (len(dataset)-step):
            temp_1 = dataset[i][0]
            temp_2 = dataset[i+step][0]

            if time_type == "date":
                #date_1 = date(temp_1)
                stamp_1 = time.mktime(datetime.strptime(temp_1, "%Y-%m-%d").timetuple())
                stamp_2 = time.mktime(datetime.strptime(temp_2, "%Y-%m-%d").timetuple())
            elif time_type == "time":
                stamp_1 = time.mktime(datetime.strptime(temp_1, "%H:%M:%S").timetuple())
                stamp_2 = time.mktime(datetime.strptime(temp_2, "%H:%M:%S").timetuple())
            else:
                stamp_1 = temp_1
                stamp_2 = temp_2

            time_diff = (stamp_2 - stamp_1)
            time_diffs.append(time_diff)
            #print(stamp_1)
    #print(time_lags)

    return time_diffs


def approx_timelag(dataset,step):
    #approximate timelag using fuzzy logic
    timelag = 0;
    return timelag;