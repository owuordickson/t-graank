"""

@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

This code automatically generates latitude and longitude (coordinates) within the Kenyan DataCube
The coordinates were uploaded to a Jupyter instance (ARDC - AWS) to fetch the NDVIs of the
generated coordinates from 2007 to 2017

"""

import decimal
import csv

LONGITUDE_L = 32.9940
LONGITUDE_R = 41.9923
LATITUDE_T = 5.9990
LATITUDE_B = -4.9990


def get_longitudes(diff):
    i = LONGITUDE_L
    lon_arr = []
    num = 1
    while i < LONGITUDE_R:
        lon = round(i,4)
        lon_arr.append(lon)
        i = i + diff
        if i >= LONGITUDE_R:
            lon_arr.append(LONGITUDE_R)
        #print(num)
        num = num + 1
    return lon_arr

def get_lattitudes(diff):
    j = LATITUDE_T
    lat_arr = []
    while j > LATITUDE_B:
        lat = round(j,4)
        lat_arr.append(lat)
        j = j - diff
        if j <= LATITUDE_B:
            lat_arr.append(LATITUDE_B)
    return lat_arr

def remove_ocean(lat,lon):
    x = (41.9923 - 38.3643)
    y = (-0.5054 - (-4.9990))
    m = x/y
    #print(m)

    if lat<-0.5054 and lon>38.3643:
        if lon >= 41.9923:
            return 0
        else:
            lon_x = (41.9923 - lon)
            lat_y = -((lon_x*m)+0.5054)#(4.9990 + lat)
            #cal_m = lon_x/lat_y
            #print(lat_y)
            if lat < lat_y:
                return 0
    return 1


def gen_macro_cells(columns, rows):
    #20 columns and 25 rows
    lon_diff = ((LONGITUDE_R - LONGITUDE_L) / columns)
    lat_diff = ((LATITUDE_T - LATITUDE_B) / rows)
    #print(lon_diff)
    #print(lat_diff)
    lons = get_longitudes(lon_diff)
    lats = get_lattitudes(lat_diff)
    # 0.2 x 0.2 degrees
    print("Latitudes:" + str(lats))
    print("Longitudes:" + str(lons))
    macro_arr = []
    for r in range(len(lats) - 1):
        for c in range(len(lons) - 1):
            macro_cell = [lats[r], lats[r + 1], lons[c], lons[c + 1]]
            macro_arr.append(macro_cell)
    #print(macro_cells)
    return macro_arr

def gen_cells(macro_cells):
    #0.2 x 0.2 degrees
    print("Total Macro Cells: "+ str(len(macro_cells)))
    #print("Macro Cells:" + str(macro_cells))
    cell_arr = [['Latitude-1','Latitude-2','Longitude-1','Longitude-2']]
    for i in range(len(macro_cells)):
        m_lat_1 = macro_cells[i][0]
        m_lat_2 = macro_cells[i][1]
        m_lon_1 = macro_cells[i][2]
        m_lon_2 = macro_cells[i][3]

        lat_center = ((m_lat_1+m_lat_2)/2)
        lon_center = ((m_lon_1+m_lon_2)/2)

        if lat_center>LATITUDE_B and lon_center<LONGITUDE_R:
            check = remove_ocean(lat_center,lon_center)
            if check == 1:
                lat_1 = round((lat_center - 0.1),4)
                lat_2 = round((lat_center + 0.1),4)
                lon_1 = round((lon_center - 0.1),4)
                lon_2 = round((lon_center + 0.1),4)

                cell = [lat_1,lat_2,lon_1,lon_2]
                cell_arr.append(cell)
    print("Total 0.2x0.2 Cells: "+str(len(cell_arr)))
    #print("0.2x0.2 Cells: "+ str(cell_arr))
    return cell_arr


def main():
    macro_cells = gen_macro_cells(15,18)

    cell_data = gen_cells(macro_cells)
    #print(cell_data)

    with open('data/kenya_jupyter.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in cell_data:
            csv_writer.writerow(data)
    csv_file.close()

main()
#n = remove_ocean(-4.9989,38.3644)
#print(n)