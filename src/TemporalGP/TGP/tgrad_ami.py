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

    def __init__(self, f_path, eq, min_sup, ref_item, err, cores):
        """"""
        # Compute MI w.r.t. RefCol with original dataset to get the actual relationship
        # between variables. Compute MI for every time-delay/time-lag: if the values are
        # almost equal to actual, then we have the most accurate time-delay. Instead of
        # min-representativity value, we propose error-margin.

        super(TGradAMI, self).__init__(f_path, eq, min_sup, ref_item=ref_item, min_rep=0.5, cores=cores)
        self.feature_cols = np.setdiff1d(self.attr_cols, self.ref_col)
        self.initial_mutual_info = self.compute_mutual_info(self.full_attr_data)
        self.error_margin = err
        self.mi_arr = list()

    def compute_mutual_info(self, attr_data):
        """"""
        y = np.array(attr_data[self.ref_col], dtype=float).T
        x_data = np.array(attr_data[self.feature_cols], dtype=float).T

        mutual_info = mutual_info_regression(x_data, y)
        # mi_series = pd.Series(mutual_info)
        # mi_series.index = self.titles[attr_cols]
        # mi_series.sort_values(ascending=False)

        return mutual_info

    def discover_tgp(self, parallel=False):
        """"""
        # 1. Compute all the MI for every time-delay and store in list
        for step in range(self.max_step):
            attr_data, time_diffs = self.transform_data(step+1)
            mi = self.compute_mutual_info(attr_data)
            self.mi_arr.append(mi)
