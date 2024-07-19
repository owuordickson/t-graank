# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Functions for estimating time-lag using AMI and extending it to mining gradual patterns.
The average mutual information I(X; Y) is a measure of the amount of “information” that
the random variables X and Y provide about one another.


"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .t_graank import TGrad


class TGradAMI(TGrad):

    def __init__(self, f_path, eq, min_sup, target_col, err, num_cores):
        """"""
        # Compute MI w.r.t. target-column with original dataset to get the actual relationship
        # between variables. Compute MI for every time-delay/time-lag: if the values are
        # almost equal to actual, then we have the most accurate time-delay. Instead of
        # min-representativity value, we propose error-margin.

        super(TGradAMI, self).__init__(f_path, eq, min_sup, target_col=target_col, min_rep=0.25, num_cores=num_cores)
        self.error_margin = err
        self.feature_cols = np.setdiff1d(self.attr_cols, self.target_col)
        self.initial_mutual_info = None
        self.mi_arr = None

    def compute_mutual_info(self):
        """"""
        # 1. Compute all the MI for every time-delay and store in list
        mi_list = []
        for step in range(self.max_step):
            attr_data, _ = self.transform_data(step)
            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[self.feature_cols], dtype=float).T
            mutual_info = mutual_info_regression(x_data, y)
            mi_list.append(mutual_info)
        mi_arr = np.array(mi_list, dtype=float)

        # 2. Standardize MI array
        # We replace 0 with -1 because 0 indicates NO MI, so we make it useless by making it -1, so it allows small
        # MI values to be considered and not 0. This is beautiful because if initial MI is 0, then both will be -1
        # making it the optimal MI with any other -1 in the time-delayed MIs
        mi_arr[mi_arr == 0] = -1
        # print(f"{mi_arr}\n")
        self.initial_mutual_info = mi_arr[0]  # step 0 is the MI without any time delay (or step)
        self.mi_arr = mi_arr[1:]

    def discover_tgp(self, parallel=False):
        """"""

        # 1. Compute mutual information
        self.compute_mutual_info()

        # 2. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(self.mi_arr, self.initial_mutual_info))
        absolute_error = np.sqrt(squared_diff)
        optimal_steps_arr = np.argmin(absolute_error, axis=0)
        max_step = (np.max(optimal_steps_arr) + 1)
        print(f"Largest step delay: {max_step}\n")
        # print(f"Initial MI: {self.initial_mutual_info}\n")
        # print(f"Delayed MIs: {self.mi_arr}\n")
        # print(f"Abs.E.: {absolute_error}\n")
        # print(f"Optimal Steps Arr: {optimal_steps_arr}\n")

        # 3. Integrate feature indices with the computed steps
        # optimal_dict = dict(map(lambda key, val: (int(key), int(val+1)), self.feature_cols, optimal_steps_arr))
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        print(f"Optimal Dict: {optimal_dict}\n")

        # 4. Create final (and dynamic) delayed dataset
        delayed_data = None
        time_dict = {}
        n = self.row_count
        k = (n - max_step)  # No. of rows created by largest step-delay
        for col_index in range(self.col_count):
            if (col_index == self.target_col) or (col_index in self.time_cols):
                # date-time column OR target column
                temp_row = self.full_attr_data[col_index][0: k]
            else:
                # other attributes
                step = optimal_dict[col_index]
                temp_row = self.full_attr_data[col_index][step: n]
                _, time_diffs = self.get_time_diffs(step)

                # Get first k items
                temp_row = temp_row[0: k]
                # time_diffs = dict(list(time_diffs.items())[0: k])
                for i in range(k):
                    if i in time_dict:
                        time_dict[i].append(time_diffs[i])
                    else:
                        time_dict[i] = [time_diffs[i]]
                # print(f"{time_diffs}\n")
                # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
            delayed_data = temp_row if (delayed_data is None) \
                else np.vstack((delayed_data, temp_row))
        # print(f"{delayed_data}\n")
        print(f"{time_dict}\n")

