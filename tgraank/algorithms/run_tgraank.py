"""
@author: "Dickson Owuor"
@copyright: "Copyright (c) 2018 Université de Montpellier"
@credits: "Anne Laurent, Joseph Orero"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"

This code fetches user preferences and executes algorithm and
returns results to the user

"""


from tgraank.algorithms.data_transform import DataTransform
from tgraank.algorithms.tgraank import *


def algorithm_init(filename,ref_item,minsup,minrep):
    try:
        # 1. Load dataset into program
        dataset = DataTransform(filename)
        #print(dataset)

        # 2. Get maximum transformation step
        max_step = dataset.get_max_step(minrep)
        #print("Transformation Step (max): "+str(step))

        # TRANSFORM DATA (for each step)
        patterns = 0
        for s in range(max_step):
            step = s+1 # because for-loop is not inclusive from range: 0 - max_step
            # 3. Calculate representativity
            chk_rep,rep_info = dataset.get_representativity(step)
            #print(rep_info)

            if chk_rep:
                # 4. Transform data
                data,time_diffs = dataset.transform_data(ref_item, step)
                #print(data)

                # 5. Execute GRAANK for each transformation
                title, D1, S1, T1 = Graank(Trad(list(data)), minsup, time_diffs, eq=False)

                pattern_found = check_for_pattern(ref_item,D1)
                if pattern_found == True:
                    print(rep_info)
                    for line in title:
                        print(line)
                    print('Pattern : Support')
                    for i in range(len(D1)):
                        # D is the Gradual Patterns, S is the support for D and T is time lag
                        if (str(ref_item+1)+'+' in D1[i]) or (str(ref_item+1)+'-' in D1[i]):
                            # select only relevant patterns w.r.t *reference item
                            print(str(D1[i]) + ' : ' + str(S1[i]) + ' | ' + str(T1[i]))
                            patterns = patterns + 1
                    print("---------------------------------------------------------")

        if patterns == 0:
            print("Oops! no relevant pattern was found")
            print("---------------------------------------------------------")

    except Exception as error:
        print(error)


def check_for_pattern(ref_item, R):
    pr = 0
    for i in range(len(R)):
        # D is the Gradual Patterns, S is the support for D and T is time lag
        if (str(ref_item + 1) + '+' in R[i]) or (str(ref_item + 1) + '-' in R[i]):
            # select only relevant patterns w.r.t *reference item
            pr = pr + 1
    if pr > 0:
        return True
    else:
        return False


def main(filename, ref_item, minsup, minrep):
    algorithm_init(filename,ref_item,minsup,minrep)
