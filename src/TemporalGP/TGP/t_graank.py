# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Description: updated version that uses aco-graank and parallel multiprocessing
"""

import gc
import numpy as np
import multiprocessing as mp
from so4gp import DataGP as Dataset
from so4gp import GI, GP, TimeLag

from .fuzzy_mf import calculate_time_lag
# from ..common.dataset import Dataset
# from ..common.gp import GI, GP, TGP


class TGP(GP):

    def __init__(self, gp=GP(), t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    def set_time_lag(self, t_lag):
        self.time_lag = t_lag


class TGrad:

    def __init__(self, f_path, eq, ref_item, min_sup, min_rep, cores):
        # For tgraank
        # self.d_set = d_set
        self.d_set = Dataset(f_path, min_sup=min_sup, eq=eq)
        cols = self.d_set.time_cols
        if len(cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.time_cols = cols
            self.min_sup = min_sup
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            self.orig_attr_data = self.d_set.data.copy().T
            self.cores = cores
        else:
            print("Dataset Error")
            self.time_ok = False
            self.time_cols = []
            raise Exception('No date-time datasets found')

    def get_max_step(self, min_rep):  # optimized
        all_rows = len(self.d_set.data)
        return all_rows - int(min_rep * all_rows)

    def discover_tgp(self, parallel=False):
        if parallel:
            # implement parallel multi-processing
            steps = range(self.max_step)
            pool = mp.Pool(self.cores)
            patterns = pool.map(self.fetch_patterns, steps)
            pool.close()
            pool.join()
            return patterns
        else:
            patterns = list()
            for step in range(self.max_step):
                t_pattern = self.fetch_patterns(step)
                if t_pattern:
                    patterns.append(t_pattern)
            return patterns

    def fetch_patterns(self, step):
        step += 1  # because for-loop is not inclusive from range: 0 - max_step
        # 1. Transform datasets
        d_set = self.d_set
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute t-graank for each transformation
        d_set.update_attributes(attr_data)
        tgps = self.graank(t_diffs=time_diffs, d_set=d_set)

        if len(tgps) > 0:
            return tgps
        return False

    def transform_data(self, step):  # optimized
        # NB: Restructure dataset based on reference item
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                ref_col = self.ref_item
                if ref_col in self.time_cols:
                    msg = "Reference column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (ref_col < 0) or (ref_col >= len(self.d_set.title)):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and " + str(len(self.d_set.title) - 1)
                    raise Exception(msg)
                else:
                    # 1. Split the transpose datasets set into column-tuples
                    attr_data = self.orig_attr_data

                    # 2. Transform the datasets using (row) n+step
                    new_attr_data = list()
                    size = len(attr_data)
                    for k in range(size):
                        col_index = k
                        tuples = attr_data[k]
                        n = tuples.size
                        # temp_tuples = np.empty(n, )
                        # temp_tuples[:] = np.NaN
                        if col_index in self.time_cols:
                            # date-time attribute
                            temp_tuples = tuples[:]
                        elif col_index == ref_col:
                            # reference attribute
                            temp_tuples = tuples[0: n - step]
                        else:
                            # other attributes
                            temp_tuples = tuples[step: n]
                        # print(temp_tuples)
                        new_attr_data.append(temp_tuples)
                    return new_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_time_diffs(self, step):  # optimized
        data = self.d_set.data
        size = len(data)
        time_diffs = []
        for i in range(size):
            if i < (size - step):
                # for col in self.time_cols:
                col = self.time_cols[0]  # use only the first date-time value
                temp_1 = str(data[i][int(col)])
                temp_2 = str(data[i + step][int(col)])
                stamp_1 = Dataset.get_timestamp(temp_1)
                stamp_2 = Dataset.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                # index = tuple([i, i + step])
                # time_diffs.append([time_diff, index])
                time_diffs.append([time_diff, i])
        return True, np.array(time_diffs)

    # Delete These
    def inv(self, g_item):
        if g_item[1] == '+':
            temp = tuple([g_item[0], '-'])
        else:
            temp = tuple([g_item[0], '+'])
        return temp

    def gen_apriori_candidates(self, R, sup, n):
        res = []
        I = []
        if len(R) < 2:
            return []
        try:
            Ck = [{x[0]} for x in R]
        except TypeError:
            Ck = [set(x[0]) for x in R]

        for i in range(len(R) - 1):
            for j in range(i + 1, len(R)):
                try:
                    R_i = {R[i][0]}
                    R_j = {R[j][0]}
                    R_o = {R[0][0]}
                except TypeError:
                    R_i = set(R[i][0])
                    R_j = set(R[j][0])
                    R_o = set(R[0][0])
                temp = R_i | R_j
                invtemp = {self.inv(x) for x in temp}
                if (len(temp) == len(R_o) + 1) and (not (I != [] and temp in I)) \
                        and (not (I != [] and invtemp in I)):
                    test = 1
                    for k in temp:
                        try:
                            k_set = {k}
                        except TypeError:
                            k_set = set(k)
                        temp2 = temp - k_set
                        invtemp2 = {self.inv(x) for x in temp2}
                        if not temp2 in Ck and not invtemp2 in Ck:
                            test = 0
                            break
                    if test == 1:
                        m = R[i][1] * R[j][1]
                        t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                        if t > sup:
                            res.append([temp, m])
                    I.append(temp)
                    gc.collect()
        return res

    def graank(self, t_diffs=None, d_set=None):

        d_set = d_set
        min_sup = d_set.thd_supp

        patterns = []
        n = d_set.attr_size
        # lst_valid_gi = gen_valid_bins(d_set.invalid_bins, d_set.attr_cols)
        valid_bins = d_set.valid_bins

        while len(valid_bins) > 0:
            valid_bins = self.gen_apriori_candidates(valid_bins, min_sup, n)
            i = 0
            while i < len(valid_bins) and valid_bins != []:
                gi_tuple = valid_bins[i][0]
                bin_data = valid_bins[i][1]
                # grp = 'dataset/' + d_set.step_name + '/valid_bins/' + gi.as_string()
                # bin_data = d_set.read_h5_dataset(grp)
                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
                if sup < min_sup:
                    del valid_bins[i]
                else:
                    z = 0
                    while z < (len(patterns) - 1):
                        if set(patterns[z].get_pattern()).issubset(set(gi_tuple)):
                            del patterns[z]
                        else:
                            z = z + 1
                    if t_diffs is not None:
                        t_lag = calculate_time_lag(bin_data, t_diffs)
                        if t_lag.valid:
                            gp = GP()
                            for obj in valid_bins[i][0]:
                                gi = GI(obj[0], obj[1].decode())
                                gp.add_gradual_item(gi)
                            gp.set_support(sup)
                            tgp = TGP(gp=gp, t_lag=t_lag)
                            patterns.append(tgp)
                    else:
                        gp = GP()
                        for obj in valid_bins[i][0]:
                            gi = GI(obj[0], obj[1].decode())
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        patterns.append(gp)
                    i += 1
        if t_diffs is None:
            return d_set, patterns
        else:
            return patterns
