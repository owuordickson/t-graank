# author: Dickson Owuor

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

    ndvi_data = [['NDVI_1','NDVI_2','Month','Year']]
    for i in range(len(ndvi_1)):
        raw_time = str(ndvi_1[i][0][0])
        time = datetime.strptime(raw_time,'%Y-%m-%d')
        month = datetime.strftime(time,'%m')
        year = datetime.strftime(time,'%Y')
        ndvi_index_1 = ndvi_1[i][1]['percent_inside_threshold']
        ndvi_index_2 = ndvi_2[i][1]['percent_inside_threshold']
        ndvi_array = [ndvi_index_1,ndvi_index_2,int(month),int(year)]
        ndvi_data.append(ndvi_array)

    #print(ndvi_data)

    with open('ndvi_file.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in ndvi_data:
            csv_writer.writerow(data)
        csv_file.close()


restructure_json("karura2012_2017.json","mtKenya2012_2017.json")