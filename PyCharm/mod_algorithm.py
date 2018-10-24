"""
@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""

import csv
from datetime import datetime
from PyCharm.graank import *
import numpy as np


def test_dataset(filename):
    #test the dataset attributes: time|item_1|item_2|...|item_n
    #return true if it ok, return (list)dataset

    #retrieve dataset from file
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        temp = list(reader)
    f.close()

    raw_time = str(temp[1][0])

    #check if dataset has time in the first column
    try:
        chk_time = datetime.strptime(raw_time, '%Y-%m-%d')
    except ValueError:
        try:
            chk_time = datetime.strptime(raw_time, '%H:%M:%S')
        except ValueError:
            #print("No timestamp found")
            return False,"No timestamp found"
        else:
            #print(chk_time)
            return True,temp;
    else:
        #print(chk_time)
        return True,temp;


def split_dataset(dataset):
    #ignore first row and first column

    #get No. of columns (ignore 1st column)
    no_columns = (len(dataset[0])-1)

    #Create arrays for each gradual column item
    multi_dataset = [None]*(no_columns)
    #print(multi_dataset)
    for c in range(no_columns):
        multi_dataset[c] = []
        for i in range(1,len(dataset)):
            item = dataset[i][c+1] #because time is the first column in dataset (it is ignored)
            multi_dataset[c].append(item)

    #print(multi_dataset)
    return multi_dataset

def transform_data(ref_column,step,dataset,multi_dataset):
    #restructure dataset
    #new_dataset = []

    #LOADING TITLES
    first_row = dataset[0]

    #Creating titles without time column
    no_columns = (len(first_row) - 1)
    title_row = [None]*no_columns
    for c in range(no_columns):
        title_row[c] = first_row[c+1]

    new_dataset = [title_row]

    #Split the original dataset into gradual items
    #gradual_items = split_dataset(dataset)
    gradual_items = multi_dataset

    for j in range(len(dataset)):
        time_diff = 0
        ref_item = gradual_items[ref_column]

        if j<len(ref_item)-step:
            init_array = [ref_item[j]]

            for i in range(len(gradual_items)):
                if i<len(gradual_items) and i!=ref_column:
                    gradual_item = gradual_items[i];
                    temp = [gradual_item[j+step]]
                    temp_array = np.append(init_array,temp,axis=0)
                    init_array = temp_array
            new_dataset.append(list(init_array))
            #return new_dataset

    return new_dataset;


def get_representativity(step,dataset):
    all_rows = len(dataset)
    sel_rows = (all_rows-step)
    if sel_rows > 0:
        rep = (sel_rows/all_rows)
        info = {"Transformation n+":step, "Representativity": rep, "Selected Rows": sel_rows, "Total Rows": all_rows}
        return True,info
    else:
        return False,"Representativity is 0%"


def get_max_step(dataset,minrep):

    for i in range(len(dataset)):
        check, info = get_representativity(i+1, dataset)
        if check:
            rep = info['Representativity']
            if rep < minrep:
                return i
        else:
            return 0


def approx_timelag(dataset,step):
    #approximate timelag using fuzzy logic
    timelag = 0;
    return timelag;


def algorithm_init(filename,ref_item,minsup,minrep):

    #TEST DATASET
    chk_data, dataset = test_dataset(filename)
    if chk_data:
        #print(dataset)

        #GET MAXIMUM TRANSFORMATION STEP
        max_step = get_max_step(dataset,minrep)
        #print("Transformation Step (max): "+str(step))
        multi_dataset = split_dataset(dataset)


        #TRANSFORM DATA
        for s in range(max_step):
            step = s+1 #because for-loop is not inclusive from range: 0 - max_step
            chk_rep,rep_info = get_representativity(step, dataset)
            print(rep_info)

            if chk_rep:
                data = transform_data(ref_item, step, dataset,multi_dataset)
                #print(data)

                # execute GRAANK for each transformation - D1, S1 = Graank(Trad(dataset), supmin1, eq)
                execute_graank(list(data),minsup)

                # estimate timelag
                approx_timelag(step,dataset)
                print("---------------------------------------------------------")
    else:
        print("Error: " + dataset)


def execute_graank(dataset,minsup):
    D1, S1 = Graank(Trad(dataset), minsup, eq=False)
    #print('D1 : ' + filename1)
    print('D1 : ')
    for i in range(len(D1)):
        #D is the Gradual Patterns, and S is the support
        print(str(D1[i]) + ' : ' + str(S1[i]))


def main(filename,ref_item,minsup,minrep):
    algorithm_init(filename,ref_item,minsup,minrep)


main("ndvi_test.csv",1,0.5,0.8)