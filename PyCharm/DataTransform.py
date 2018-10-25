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
from PyCharm.ModifiedGRAANK import *


class DataSet:

    def __init__(self,dataset):
        self.dataset = dataset
        self.multi_dataset = self.split_dataset()

    def split_dataset(self):
        # ignore first row and first column

        # get No. of columns (ignore 1st column)
        no_columns = (len(self.dataset[0]) - 1)

        # Create arrays for each gradual column item
        multi_dataset = [None] * (no_columns)
        # print(multi_dataset)
        for c in range(no_columns):
            multi_dataset[c] = []
            for i in range(1, len(self.dataset)):
                item = self.dataset[i][c + 1]  # because time is the first column in dataset (it is ignored)
                multi_dataset[c].append(item)

        # print(multi_dataset)
        return multi_dataset

    def transform_data(self,ref_column, step):
        # restructure dataset
        # new_dataset = []

        # LOADING TITLES
        first_row = self.dataset[0]

        # Creating titles without time column
        no_columns = (len(first_row) - 1)
        title_row = [None] * no_columns
        for c in range(no_columns):
            title_row[c] = first_row[c + 1]

        ref_name = str(title_row[ref_column])
        title_row[ref_column] = ref_name + "**"
        new_dataset = [title_row]

        # Split the original dataset into gradual items
        # gradual_items = split_dataset(dataset)
        gradual_items = self.multi_dataset

        for j in range(len(self.dataset)):
            # time_diff = 0
            ref_item = gradual_items[ref_column]

            if j < len(ref_item) - step:
                init_array = [ref_item[j]]

                for i in range(len(gradual_items)):
                    if i < len(gradual_items) and i != ref_column:
                        gradual_item = gradual_items[i];
                        temp = [gradual_item[j + step]]
                        temp_array = np.append(init_array, temp, axis=0)
                        init_array = temp_array
                new_dataset.append(list(init_array))
                # return new_dataset

        return new_dataset;

    def get_representativity(self, step):
        all_rows = (len(self.dataset) - 1)  # removing the title row
        sel_rows = (all_rows - step)
        if sel_rows > 0:
            rep = (sel_rows / all_rows)
            info = {"Transformation n+": step, "Representativity": rep, "Selected Rows": sel_rows,
                    "Total Rows": all_rows}
            return True, info
        else:
            return False, "Representativity is 0%"

    def get_max_step(self, minrep):

        for i in range(len(self.dataset)):
            check, info = self.get_representativity(i + 1)
            if check:
                rep = info['Representativity']
                if rep < minrep:
                    return i
            else:
                return 0


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
                return False,None,"No timestamp found"
            else:
                #print(chk_time)
                return True,"time",temp;
        else:
            #print(chk_time)
            return True,"date",temp;