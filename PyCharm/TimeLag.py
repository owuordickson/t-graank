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
import statistics
import skfuzzy as fuzzy
import numpy as np


def get_time_diffs(dataset,step):

    time_diffs = []

    for i in range(1,len(dataset.data)):
        if i < (len(dataset.data)-step):
            temp_1 = dataset.data[i][0]
            temp_2 = dataset.data[i+step][0]

            #to be changed to a more thorough function
            if dataset.time_type == "date":
                stamp_1 = time.mktime(datetime.strptime(temp_1, "%Y-%m-%d").timetuple())
                stamp_2 = time.mktime(datetime.strptime(temp_2, "%Y-%m-%d").timetuple())
            elif dataset.time_type == "time":
                stamp_1 = time.mktime(datetime.strptime(temp_1, "%H:%M:%S").timetuple())
                stamp_2 = time.mktime(datetime.strptime(temp_2, "%H:%M:%S").timetuple())
            else:
                stamp_1 = temp_1
                stamp_2 = temp_2

            time_diff = (stamp_2 - stamp_1)
            time_diffs.append(time_diff)

    #print(time_lags)
    return time_diffs


def approx_timelag(time_lags,dataset,minsup,step):
    #approximate timelag using fuzzy logic

    #1. Get time differences
    time_diffs = get_time_diffs(dataset,step)

    #2. Sort the time lags
    time_diffs.sort()

    #3. Get the boundaries of membership function
    min = time_diffs[0] #to be changed to quartile 1
    med = statistics.median(time_diffs)
    max = time_diffs[(len(time_diffs)-1)] #to be changed to quartile 3
    boundaries = [min,med,max]
    #print(boundaries)

    #4. Calculate membership of frequent path
    memberships = fuzzy.membership.trimf(np.array(time_lags),np.array(boundaries))
    #print(memberships)

    #5. Calculate support
    sup = calculate_support(memberships)

    if sup >= minsup:
        return med,sup
    else:
        #6. Slide to the left to change boundaries
        print()
        #7. Slide to the right to change boundaries

        #8. Expand quartiles and repeat 5. and 6.

    return med,sup


def calculate_support(memberships):

    sup_count = 0
    total = len(memberships)
    for i in range(total):
        if float(memberships[i]) > 0.5:
            sup_count = sup_count + 1
    support = sup_count / total

    #print("Support Count: " + str(sup_count))
    #print("Total Count: " + str(len(memberships)))
    #print("Support: "+ str(support))

    return support
