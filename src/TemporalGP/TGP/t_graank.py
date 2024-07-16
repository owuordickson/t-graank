# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Algorithm for mining temporal gradual patterns using fuzzy membership functions.
"""


import numpy as np
import skfuzzy as fuzzy
import multiprocessing as mp
from so4gp import DataGP as Dataset
from so4gp import GI, GP, TimeLag, GRAANK


class TGP(GP):

    def __init__(self, gp=GP(), t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    def set_time_lag(self, t_lag):
        self.time_lag = t_lag


class TGrad(GRAANK):

    def __init__(self, f_path, eq, min_sup, ref_item, min_rep, cores):
        """"""

        super(TGrad, self).__init__(data_source=f_path, min_sup=min_sup, eq=eq)
        if len(self.time_cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.ref_item = ref_item
            self.max_step = self.get_max_step(min_rep)
            self.orig_attr_data = self.data.copy().T
            self.cores = cores
        else:
            print("Dataset Error")
            self.time_ok = False
            raise Exception('No date-time datasets found')

    def get_max_step(self, min_rep):
        """"""
        return self.row_count - int(min_rep * self.row_count)

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
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute t-graank for each transformation
        t_gps = self.discover(t_diffs=time_diffs, attr_data=attr_data)

        if len(t_gps) > 0:
            return t_gps
        return False

    def transform_data(self, step):
        """"""
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
                elif (ref_col < 0) or (ref_col >= self.col_count):
                    msg = "Reference column does not exist\nselect column between: " \
                          "0 and " + str(self.col_count - 1)
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
        """"""
        size = self.row_count
        time_diffs = []
        for i in range(size):
            if i < (size - step):
                # for col in self.time_cols:
                col = self.time_cols[0]  # use only the first date-time value
                temp_1 = str(self.data[i][int(col)])
                temp_2 = str(self.data[i + step][int(col)])
                stamp_1 = TGrad.get_timestamp(temp_1)
                stamp_2 = TGrad.get_timestamp(temp_2)
                if (not stamp_1) or (not stamp_2):
                    return False, [i + 1, i + step + 1]
                time_diff = (stamp_2 - stamp_1)
                # index = tuple([i, i + step])
                # time_diffs.append([time_diff, index])
                time_diffs.append([time_diff, i])
        return True, np.array(time_diffs)

    def discover(self, t_diffs=None, attr_data=None):
        """"""

        self.fit_bitmap(attr_data)

        gradual_patterns = []
        """:type gradual_patterns: list"""
        n = self.attr_size
        valid_bins = self.valid_bins
        # n = d_set.attr_size
        # valid_bins = d_set.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins)
            invalid_count += inv_count
            i = 0
            while i < len(valid_bins) and valid_bins != []:
                gi_tuple = valid_bins[i][0]
                bin_data = valid_bins[i][1]
                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
                if sup < self.thd_supp:
                    del valid_bins[i]
                    invalid_count += 1
                else:
                    z = 0
                    while z < (len(gradual_patterns) - 1):
                        if set(gradual_patterns[z].get_pattern()).issubset(set(gi_tuple)):
                            del gradual_patterns[z]
                        else:
                            z = z + 1

                    t_lag = TGrad.get_fuzzy_time_lag(bin_data, t_diffs)
                    if t_lag.valid:
                        gp = GP()
                        for obj in valid_bins[i][0]:
                            gi = GI(obj[0], obj[1].decode())
                            """:type gi: GI"""
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        tgp = TGP(gp=gp, t_lag=t_lag)
                        gradual_patterns.append(tgp)
                    i += 1

        return gradual_patterns

    @staticmethod
    def get_timestamp(time_data):
        """"""
        try:
            ok, stamp = Dataset.test_time(time_data)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False

    @staticmethod
    def get_fuzzy_time_lag(bin_data, time_diffs):
        # 1. Get Indices
        indices = np.argwhere(bin_data == 1)

        # 2. Get TimeLags
        pat_indices_flat = np.unique(indices.flatten())
        time_lags = list()
        for obj in time_diffs:
            index1 = obj[1]
            if int(index1) in pat_indices_flat:
                time_lags.append(obj[0])
        time_lags = np.array(time_lags)

        # 3. Approximate TimeLag using Fuzzy Membership
        time_lag = TGrad.__approximate_fuzzy_time_lag__(time_lags)
        return time_lag

    @staticmethod
    def __approximate_fuzzy_time_lag__(time_lags):
        if len(time_lags) <= 0:
            # if time_lags is blank return nothing
            return TimeLag()
        else:
            time_lags = np.absolute(np.array(time_lags))
            min_a = np.min(time_lags)
            max_c = np.max(time_lags)
            count = time_lags.size + 3
            tot_boundaries = np.linspace(min_a / 2, max_c + 1, num=count)

            sup1 = 0
            center = time_lags[0]
            size = len(tot_boundaries)
            for i in range(0, size, 2):
                if (i + 3) <= size:
                    boundaries = tot_boundaries[i:i + 3:1]
                else:
                    boundaries = tot_boundaries[size - 3:size:1]
                memberships = fuzzy.membership.trimf(time_lags, boundaries)

                # Compute Support
                sup_count = np.count_nonzero(memberships > 0)
                total = memberships.size
                sup = sup_count / total
                # sup = calculate_support(memberships)

                if sup > sup1:
                    sup1 = sup
                    center = boundaries[1]
                if sup >= 0.5:
                    return TimeLag(boundaries[1], sup)
            return TimeLag(center, sup1)
