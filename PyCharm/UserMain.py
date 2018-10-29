"""
@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

This code fetched user preferences and executes algorithm and returns results

"""


from PyCharm.algorithm.DataTransform import DataSet
from PyCharm.algorithm.TimeLag import approx_timelag
from PyCharm.algorithm.ModifiedGRAANK import *

def algorithm_init(filename,ref_item,minsup,minrep):

    try:
        #1. Load dataset into program
        dataset = DataSet(filename)
        #print(dataset)

        #2. Get maximum transformation step
        max_step = dataset.get_max_step(minrep)
        #print("Transformation Step (max): "+str(step))


        #TRANSFORM DATA (for each step)
        for s in range(max_step):
            step = s+1 #because for-loop is not inclusive from range: 0 - max_step
            #3. Calculate representativity
            chk_rep,rep_info = dataset.get_representativity(step)
            print(rep_info)

            if chk_rep:
                #4. Transform data
                data = dataset.transform_data(ref_item, step)
                #print(data)

                #5. Execute GRAANK for each transformation
                D1, S1, I1 = Graank(Trad(list(data)), minsup, eq=False)
                print('Pattern : Support')
                for i in range(len(D1)):
                    # D is the Gradual Patterns, and S is the support
                    print(str(D1[i]) + ' : ' + str(S1[i]))

                    #6. Estimate time lag
                    sup_msg = approx_timelag(I1, dataset, minsup, step)
                    print(sup_msg)

                print("---------------------------------------------------------")
    except Exception as error:
        print(error)


def main(filename,ref_item,minsup,minrep):
    algorithm_init(filename,ref_item,minsup,minrep)


main("data/ndvi_test.csv",0,0.5,0.8)
#main("test.csv",0,0.5,0.6)