"""
@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

"""


from PyCharm.DataTransform import *
from PyCharm.TimeLag import *

def algorithm_init(filename,ref_item,minsup,minrep):

    #TEST DATASET
    chk_data, time_type, dataset = test_dataset(filename)
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

                #Execute GRAANK for each transformation - D1, S1 = Graank(Trad(dataset), supmin1, eq)
                D1, S1 = Graank(Trad(list(data)), minsup, eq=False)
                print('Pattern : Support')
                for i in range(len(D1)):
                    # D is the Gradual Patterns, and S is the support
                    print(str(D1[i]) + ' : ' + str(S1[i]))

                # estimate timelag
                approx_timelag(step,dataset)

                print("---------------------------------------------------------")
    else:
        print("Error: " + dataset)


def main(filename,ref_item,minsup,minrep):
    #algorithm_init(filename,ref_item,minsup,minrep)
    chk_data, time_type, dataset = test_dataset(filename)
    #print(dataset)
    print(get_time_diffs(dataset,time_type,step=3))


main("ndvi_test.csv",0,0.5,0.8)