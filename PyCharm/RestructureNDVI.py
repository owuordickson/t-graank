
__author__ = "Dickson Owuor"
__copyright__ = "Copyright (c) 2018 Université de Montpellier"
__credits__ = ["Anne Laurent", "Joseph Orero"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "owuordickson@gmail.com"

#This code converts NDVI json data into a csv format for pattern mining


import json
import csv
import numpy as np
from datetime import datetime


def fetch_data(filename):
    with open(filename, mode='r') as json_file:
        data_array = json.load(json_file)
        json_file.close()
    return data_array


def save_to_csv(new_file,new_data):
    with open(new_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in new_data:
            csv_writer.writerow(data)
        csv_file.close()


def restructure_data_1(filename):
    raw_data = fetch_data(filename)

    #setting up the titles
    #ndvi_data = []
    init_arr = ['Date']
    for i in range(0,len(raw_data),30): #5):
        temp = [raw_data[i]['Area']]
        temp_arr = np.append(init_arr, temp, axis=0)
        init_arr = temp_arr
    ndvi_data = [init_arr]
    #print(ndvi_data)

    #extracting all NDVI indices and dates of all areas
    temp_data = []
    for i in range(0,len(raw_data),30): #5):
        ndvi_arr = raw_data[i]['Data']
        temp_arr = []
        for j in range(len(ndvi_arr)):
            raw_time = str(ndvi_arr[j][0][0])
            ndvi_index = ndvi_arr[j][1]['percent_inside_threshold']
            temp_arr.append([raw_time,ndvi_index])
        temp_data.append(temp_arr)
    #print(temp_data)

    #formating NDVI in raw format
    for i in range(len(temp_data[0])): #rows
        init_arr = []
        for j in range(len(temp_data)): #columns
            if j == 0:
                init_arr = [temp_data[j][i][0]] #time value
            temp = temp_data[j][i][1] #index value
            if int(temp) == 200:
                init_arr = []
                break
            temp_arr = np.append(init_arr, [temp], axis=0)
            init_arr = temp_arr
        #print(temp_arr)
        if len(init_arr) > 0:
            ndvi_data.append(init_arr)

    #ndvi_data = np.array(ndvi_data)
    #print(ndvi_data)
    save_to_csv("data/ndvi_kenya.csv", ndvi_data)


def restructure_data_2(filename_1,filename_2):
    ndvi_1 = fetch_data(filename_1)['Data']
    ndvi_2 = fetch_data(filename_2)['Data']

    # ndvi data is structured without factoring in time (same dates)
    ndvi_data = [['Date','NDVI_1(karura)', 'NDVI_2(mt_kenya)']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
        ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']

        #the whole data in csv
        ndvi_array = [raw_time, ndvi_index_1, ndvi_index_2]
        ndvi_data.append(ndvi_array)

    print("Data Set \n")
    print(ndvi_data)
    save_to_csv("data/ndvi_test.csv",ndvi_data)


#restructure_data_2("data/karura2012_2017.json","data/mtKenya2012_2017.json")
restructure_data_1("data/kenya2007_2017.json")