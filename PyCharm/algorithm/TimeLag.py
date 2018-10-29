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
import skfuzzy as fuzzy
import numpy as np
from scipy import stats


def get_time_diffs(dataset,step):

    time_diffs = []

    for i in range(1,len(dataset.data)):
        if i < (len(dataset.data)-step):
            temp_1 = dataset.data[i][0]
            temp_2 = dataset.data[i+step][0]

            stamp_1 = get_timestamp(temp_1)
            stamp_2 = get_timestamp(temp_2)

            if stamp_1==False or stamp_2==False:
                return False,[i+1,i+step+1]

            time_diff = (stamp_2 - stamp_1)
            time_diffs.append(time_diff)

    #print(time_lags)
    return True,time_diffs


def get_time_lags(indices,dataset):

    indxs = get_unique_index(indices)

    time_lags = []
    for index in indxs:
        r1 = index[0]+1 #including the title row
        r2 = index[1]+1

        if r1>r2:
            temp_1 = dataset.data[r2][0]
            temp_2 = dataset.data[r1][0]
        else:
            temp_1 = dataset.data[r1][0]
            temp_2 = dataset.data[r2][0]

        stamp_1 = get_timestamp(temp_1)
        stamp_2 = get_timestamp(temp_2)

        time_lag = (stamp_2 - stamp_1)
        time_lags.append(time_lag)

    #print(time_lags)
    return time_lags

def get_unique_index(indices):

    indxs = []
    if len(indices)>0:
        inds = indices[0]
        #print(inds)
        for i in range(len(inds)):
            index = inds[i]
            r = index[0]
            c = index[1]
            if not indxs:
                indxs.append([r+1, c+1])
            #elif r != inds[i - 1][0]: #for unique concordant pairs
            else:#returns all concordant pairs
                r = index[0]
                c = index[1]
                indxs.append([r+1, c+1])

    #print(indxs)
    return indxs

def approx_timelag(indices,dataset,minsup,step):
    #approximate timelag using fuzzy logic

    if dataset.time_ok:

        #1. Get time differences
        ok,time_diffs = get_time_diffs(dataset,step)
        if ok==False:
            msg = "Time in row "+ str(time_diffs[0])+" or row "+str(time_diffs[1])+" is not valid."
            return msg

        #2. Sort the time lags
        time_diffs.sort()
        #print(time_diffs)

        #3. Get the boundaries of membership function
        min = np.min(time_diffs) #to be changed to quartile 1
        q_1 = np.percentile(time_diffs, 25)  # Q1
        med = np.percentile(time_diffs, 50)
        q_3 = np.percentile(time_diffs, 75)
        max = np.max(time_diffs) #to be changed to quartile 3
        boundaries = [q_1,med,q_3]
        extremes = [min,max]
        #print(boundaries)

        #4. Get time lags for the path
        time_lags = get_time_lags(indices,dataset)
        #print(time_lags)

        time_lag,sup = optimize_timelag(minsup,time_lags,boundaries,extremes)

        if sup != False:
            lag_msg = get_time_lag_format(time_lag)
            msg = (lag_msg + " (Support: " + str(sup) + ")")
            return msg
        else:
            msg = "Unable to estimate time lag"
            return msg
    else:
        msg = "Time format in 1st column could not be processed"
        return msg


def optimize_timelag(minsup,timelags,orig_boundaries,extremes):

    boundaries = orig_boundaries
    slice = (0.1*int(orig_boundaries[1]))
    sup = sup1 = 0
    slide_left = slide_right = expand = False
    #sample = np.percentile(timelags, 50)
    mode = stats.mode(timelags)
    sample = int(mode[0])

    a = boundaries[0]
    b = b1 = boundaries[1]
    c = boundaries[2]
    min_a = extremes[0]
    max_c = extremes[1]
    #print(mode)

    while(sup <= minsup):

        if sup > sup1:
            sup1 = sup
            b1 = b

        # Calculate membership of frequent path
        memberships = fuzzy.membership.trimf(np.array(timelags), np.array(boundaries))
        # print(memberships)

        # Calculate support
        sup = calculate_support(memberships)
        #print("Support"+str(sup))

        if sup >= minsup:
            return b,sup
        else:
            if slide_left == False:
                # 7. Slide to the left to change boundaries
                #if extreme is reached - then slide right
                if sample <= b:
                    #print("left: "+str(b))
                    a = a - slice
                    b = b - slice
                    c = c - slice
                    boundaries = [a,b,c]
                else:
                    slide_left = True
            elif slide_right == False:
                # 8. Slide to the right to change boundaries
                # if extreme is reached - then slide right
                if sample >= b:
                    #print("right: "+str(b))
                    a = a + slice
                    b = b + slice
                    c = c + slice
                    boundaries = [a, b, c]
                else:
                    slide_right = True
            elif expand == False:
                # 9. Expand quartiles and repeat 5. and 6.
                a = min_a
                b = orig_boundaries[1]
                c = max_c
                boundaries = [a, b, c]
                slide_left = slide_right = False
                expand = True
                #print("expand: " + str(b))
            else:
                return b1,sup1


def calculate_support(memberships):

    #print(memberships)
    support = 0
    if len(memberships)>0:
        sup_count = 0
        total = len(memberships)
        for i in range(total):
            if float(memberships[i]) > 0.5:
                sup_count = sup_count + 1
        support = sup_count / total

    return support


def get_timestamp(time_data):

    try:
        ok,stamp = test_time(time_data)
    except ValueError:
        return False
    else:
        return stamp


def test_time(time_data):

    #add all the possible formats
    time_formats = ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y',"%H:%M:%S")

    for fmt in time_formats:
        try:
            return True,time.mktime(datetime.strptime(time_data, fmt).timetuple())
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def get_time_lag_format(median):

    if median < 0:
        sign = "-"
    else:
        sign = "+"

    t_lag, t_type = round_time(abs(median))
    msg = ("~ " + sign + str(t_lag) + " " + str(t_type))
    return msg


def round_time(seconds):

    years = seconds/3.154e+7
    months = seconds/2.628e+6
    weeks = seconds/604800
    days = seconds/86400
    hours = seconds/3600
    minutes = seconds/60

    if int(years) <= 0:
        if int(months) <= 0:
            if int(weeks) <= 0:
                if int(days) <= 0:
                    if int(hours) <= 0:
                        if int(minutes) <= 0:
                            return seconds,"seconds"
                        else:
                            return minutes,"minutes"
                    else:
                        return hours,"hours"
                else:
                    return days,"days"
            else:
                return weeks,"weeks"
        else:
            return months,"months"
    else:
        return years,"years"