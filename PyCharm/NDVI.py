
__author__ = "Dickson Owuor"
__copyright__ = "Copyright (c) 2018 Université de Montpellier"
__credits__ = ["Anne Laurent", "Joseph Orero"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "owuordickson@gmail.com"

#This code converts NDVI json data into a csv format for pattern mining


import json
import csv
from datetime import datetime

def fetchData(filename):

    with open(filename, mode='r') as json_file:
        data_array = json.load(json_file)
        json_file.close()
    return(data_array['Data'])


def restructure_json(filename_1,filename_2):

    ndvi_1 = fetchData(filename_1)
    ndvi_2 = fetchData(filename_2)

    #ndvi data is structured without factoring in time (same dates)
    ndvi_data_1 = [['NDVI_1','NDVI_2','Month','Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time,'%Y-%m-%d')
        month = datetime.strftime(time,'%m')
        year = datetime.strftime(time,'%Y')
        ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
        ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']
        if ndvi_index_1 <= 100 and ndvi_index_2 <= 100:
            ndvi_array = [ndvi_index_1,ndvi_index_2,int(month),int(year)]
            ndvi_data_1.append(ndvi_array)

    print("\n\n First Data Set \n\n")
    print(ndvi_data_1)

    #ndvi data is structured to arrange ndvi(2) 3 months later
    ndvi_data_2 = [['NDVI_1','NDVI_2(+3months)','Month','Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time,'%Y-%m-%d')
        month = datetime.strftime(time,'%m')
        year = datetime.strftime(time,'%Y')
        if i < (len(ndvi_1) - 1):
            ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
            ndvi_index_2 = ndvi_2[i+1][1]['percent_inside_threshold']
            if ndvi_index_1 <= 100 and ndvi_index_2 <= 100:
                ndvi_array = [ndvi_index_1, ndvi_index_2, int(month), int(year)]
                ndvi_data_2.append(ndvi_array)

    print("\n\n Second Data Set \n\n")
    print(ndvi_data_2)

    # ndvi data is structured to arrange ndvi(2) 6 months later
    ndvi_data_3 = [['NDVI_1', 'NDVI_2(+6months)', 'Month', 'Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time, '%Y-%m-%d')
        month = datetime.strftime(time, '%m')
        year = datetime.strftime(time, '%Y')
        if i < (len(ndvi_1) - 2):
            ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
            ndvi_index_2 = ndvi_2[i + 2][1]['percent_inside_threshold']
            if ndvi_index_1 <= 100 and ndvi_index_2 <= 100:
                ndvi_array = [ndvi_index_1, ndvi_index_2, int(month), int(year)]
                ndvi_data_3.append(ndvi_array)

    print("\n\n Third Data Set \n\n")
    print(ndvi_data_3)

    # ndvi data is structured to arrange ndvi(1) 3 months later
    ndvi_data_4 = [['NDVI_1(+3months)', 'NDVI_2', 'Month', 'Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time, '%Y-%m-%d')
        month = datetime.strftime(time, '%m')
        year = datetime.strftime(time, '%Y')
        if i < (len(ndvi_1) - 1):
            ndvi_index_1 = ndvi_1[i + 1][1]['percent_inside_threshold']
            ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']
            if ndvi_index_1 <= 100 and ndvi_index_2 <= 100:
                ndvi_array = [ndvi_index_1, ndvi_index_2, int(month), int(year)]
                ndvi_data_4.append(ndvi_array)

    print("\n\n Fourth Data Set \n\n")
    print(ndvi_data_4)

    # ndvi data is structured to arrange ndvi(1) 6 months later
    ndvi_data_5 = [['NDVI_1(+6months)', 'NDVI_2', 'Month', 'Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time, '%Y-%m-%d')
        month = datetime.strftime(time, '%m')
        year = datetime.strftime(time, '%Y')
        if i < (len(ndvi_1) - 2):
            ndvi_index_1 = ndvi_1[i + 2][1]['percent_inside_threshold']
            ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']
            if ndvi_index_1 <= 100 and ndvi_index_2 <=100:
                ndvi_array = [ndvi_index_1, ndvi_index_2, int(month), int(year)]
                ndvi_data_5.append(ndvi_array)

    print("\n\n Fifth Data Set \n\n")
    print(ndvi_data_5)

    with open('PyCharm/ndvi_file.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in ndvi_data_4:
            csv_writer.writerow(data)
        csv_file.close()


def restructure_data(filename_1,filename_2):
    ndvi_1 = fetchData(filename_1)
    ndvi_2 = fetchData(filename_2)

    # ndvi data is structured without factoring in time (same dates)
    #ndvi_data = [['Year','Month','NDVI_1', 'NDVI_2','NDVI_1(+3months)','NDVI_2(+3months)','NDVI_1(+6months)','NDVI_2(+6months)']]
    #ndvi_data = [[ 'NDVI_1', 'NDVI_2', 'NDVI_1(+3months)', 'NDVI_2(+3months)', 'NDVI_1(+6months)','NDVI_2(+6months)']]
    ndvi_data = [['Date','NDVI_1(karura)', 'NDVI_2(mt_kenya)']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time, '%Y-%m-%d')
        #month = datetime.strftime(time, '%m')
        #year = datetime.strftime(time, '%Y')
        ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
        ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']

        #the whole data in csv
        ndvi_array = [raw_time, ndvi_index_1, ndvi_index_2]
        ndvi_data.append(ndvi_array)

        #if i < (len(ndvi_1) - 1):
            #ndvi_index_1_3 = ndvi_1[i + 1][1]['percent_inside_threshold']
            #ndvi_index_2_3 = ndvi_2[i + 1][1]['percent_inside_threshold']
        #if i < (len(ndvi_1) - 2):
            #ndvi_index_1_6 = ndvi_1[i + 2][1]['percent_inside_threshold']
            #ndvi_index_2_6 = ndvi_2[i + 2][1]['percent_inside_threshold']

            #if ndvi_index_1 <= 100 and ndvi_index_2 <= 100:
                #ndvi_array = [int(year), int(month),ndvi_index_1,ndvi_index_2,ndvi_index_1_3,ndvi_index_2_3,ndvi_index_1_6,ndvi_index_2_6]
                #ndvi_array = [ndvi_index_1, ndvi_index_2, ndvi_index_1_3, ndvi_index_2_3,ndvi_index_1_6, ndvi_index_2_6]
                #ndvi_data.append(ndvi_array)

    print("Data Set \n")
    print(ndvi_data)

    with open('PyCharm/ndvi_test.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in ndvi_data:
            csv_writer.writerow(data)
        csv_file.close()


restructure_data("PyCharm/karura2012_2017.json","PyCharm/mtKenya2012_2017.json")