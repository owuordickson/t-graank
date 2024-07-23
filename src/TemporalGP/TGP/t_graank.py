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
from so4gp import GI, ExtGP, TimeLag, GRAANK


class TGP(ExtGP):

    def __init__(self, gp=ExtGP(), t_lag=TimeLag()):
        super().__init__()
        self.gradual_items = gp.gradual_items
        self.support = gp.support
        self.time_lag = t_lag

    def set_time_lag(self, t_lag):
        self.time_lag = t_lag

    @staticmethod
    def remove_subsets(gp_list, gi_arr):
        """
        Description

        Remove subset GPs from the list.

        :param gp_list: list of existing GPs
        :type gp_list: list[so4gp.ExtGP]

        :param gi_arr: gradual items in an array
        :type gi_arr: set

        :return: list of GPs
        """
        mod_gp_list = []
        for gp in gp_list:
            result1 = set(gp.get_pattern()).issubset(gi_arr)
            result2 = set(gp.inv_pattern()).issubset(gi_arr)
            if not (result1 or result2):
                mod_gp_list.append(gp)

        return mod_gp_list


class TGrad(GRAANK):

    def __init__(self, f_path: str, eq: bool, min_sup: float, target_col: int, min_rep: float, num_cores: int):
        """"""

        super(TGrad, self).__init__(data_source=f_path, min_sup=min_sup, eq=eq)
        if len(self.time_cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            self.target_col = target_col
            self.max_step = self.row_count - int(min_rep * self.row_count)
            self.full_attr_data = self.data.copy().T
            self.cores = num_cores
        else:
            print("Dataset Error")
            self.time_ok = False
            raise Exception('No date-time datasets found')

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
                t_gps = self.fetch_patterns(step + 1)  # because for-loop is not inclusive from range: 0 - max_step
                if t_gps:
                    patterns.append(t_gps)
            return patterns

    def fetch_patterns(self, step):
        # 1. Transform datasets
        attr_data, time_diffs = self.transform_data(step)

        # 2. Execute t-graank for each transformation
        t_gps = self.discover(t_diffs=time_diffs, attr_data=attr_data)

        if len(t_gps) > 0:
            return t_gps
        return False

    def transform_data(self, step):
        """"""
        # NB: Restructure dataset based on target/reference col
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                tgt_col = self.target_col
                if tgt_col in self.time_cols:
                    msg = "Target column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (tgt_col < 0) or (tgt_col >= self.col_count):
                    msg = "Target column does not exist\nselect column between: " \
                          "0 and " + str(self.col_count - 1)
                    raise Exception(msg)
                else:
                    delayed_attr_data = None
                    n = self.row_count
                    for col_index in range(self.col_count):
                        # Transform the datasets using (row) n+step
                        if (col_index == tgt_col) or (col_index in self.time_cols):
                            # date-time column OR target column
                            temp_row = self.full_attr_data[col_index][0: (n - step)]
                        else:
                            # other attributes
                            temp_row = self.full_attr_data[col_index][step: n]

                        delayed_attr_data = temp_row if (delayed_attr_data is None) \
                            else np.vstack((delayed_attr_data, temp_row))
                    # print(f"Time Diffs: {time_diffs}\n")
                    # print(f"{self.full_attr_data}: {type(self.full_attr_data)}\n")
                    # print(f"{delayed_attr_data}: {type(delayed_attr_data)}\n")
                    return delayed_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_time_diffs(self, step):  # optimized
        """"""
        size = self.row_count
        time_diffs = {}
        for i in range(size):
            if i < (size - step):
                stamp_1 = 0
                stamp_2 = 0
                for col in self.time_cols:  # sum timestamps from all time-columns
                    temp_1 = str(self.data[i][int(col)])
                    temp_2 = str(self.data[i + step][int(col)])
                    temp_stamp_1 = TGrad.get_timestamp(temp_1)
                    temp_stamp_2 = TGrad.get_timestamp(temp_2)
                    if (not temp_stamp_1) or (not temp_stamp_2):
                        # Unable to read time
                        return False, [i + 1, i + step + 1]
                    else:
                        stamp_1 += temp_stamp_1
                        stamp_2 += temp_stamp_2
                time_diff = (stamp_2 - stamp_1)
                if time_diff < 0:
                    # Error time CANNOT go backwards
                    return False, [i + 1, i + step + 1]
                time_diffs[int(i)] = float(time_diff)
        return True, time_diffs

    def discover(self, t_diffs=None, attr_data=None):
        """"""

        self.fit_bitmap(attr_data)

        gradual_patterns = []
        """:type gradual_patterns: list"""
        n = self.attr_size
        valid_bins = self.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins, self.target_col)
            invalid_count += inv_count
            i = 0
            while i < len(valid_bins) and valid_bins != []:
                gi_arr = valid_bins[i][0]
                bin_data = valid_bins[i][1]
                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
                if sup < self.thd_supp:
                    del valid_bins[i]
                    invalid_count += 1
                else:
                    # Remove subsets
                    gradual_patterns = TGP.remove_subsets(gradual_patterns, set(gi_arr))

                    t_lag = TGrad.get_fuzzy_time_lag(bin_data, t_diffs)
                    if t_lag.valid:
                        gp = ExtGP()
                        """:type gp: ExtGP"""
                        for obj in gi_arr:
                            gi = GI(obj[0], obj[1].decode())
                            """:type gi: GI"""
                            gp.add_gradual_item(gi)
                        gp.set_support(sup)
                        tgp = TGP(gp=gp, t_lag=t_lag)
                        """:type tgp: TGP"""
                        gradual_patterns.append(tgp)
                    # else:
                    #    print(f"{t_lag.timestamp} - {gi_arr}")
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
        for row, stamp_diff in time_diffs.items():
            if int(row) in pat_indices_flat:
                time_lags.append(stamp_diff)
        time_lags = np.array(time_lags)

        # 3. Approximate TimeLag using Fuzzy Membership
        time_lag = TGrad.__approximate_fuzzy_time_lag__(time_lags)
        return time_lag

    @staticmethod
    def triangular_mf(x, a, b, c):
        """"""
        if a <= x <= b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return (c - x) / (c - b)
        else:
            return 0

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
