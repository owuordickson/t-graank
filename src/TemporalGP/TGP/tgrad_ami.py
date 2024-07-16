# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Functions for estimating time-lag using AMI and extending it to mining gradual patterns.
The average mutual information I(X; Y) is a measure of the amount of “information” that the random variables X and Y
provide about one another.


"""


from t_graank import TGrad


class TGradAMI(TGrad):

    def __init__(self):
        """"""
        # Compute MI w.r.t. RefCol with original dataset to get the actual relationship between variables. Compute MI
        # for every time-delay/time-lag: if the values are almost equal to actual, then we have the most accurate
        # time-delay. Instead of min-representativity value, we propose error-margin.
        # super(TGradAMI, self).__init__(f_path=f_path, min_sup=min_sup, eq=eq)
        pass
