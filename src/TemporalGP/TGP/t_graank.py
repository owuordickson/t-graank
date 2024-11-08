# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Algorithm for mining temporal gradual patterns using fuzzy membership functions.
"""


import json
import numpy as np
import skfuzzy as fuzzy
import multiprocessing as mp

from so4gp import DataGP, GI, TGP, TimeDelay, GRAANK
from .tgrad_ami import TGradAMI


class TGrad(GRAANK):
    """Description of class TGrad.

    TGrad is an algorithm that is used to extract temporal gradual patterns from numeric datasets. An algorithm for
    mining temporal gradual patterns using fuzzy membership functions. It uses technique published
    in: https://ieeexplore.ieee.org/abstract/document/8858883.

    """

    def __init__(self, *args, target_col: int, min_rep: float = 0.5):
        """
        TGrad is an algorithm that is used to extract temporal gradual patterns from numeric datasets.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param target_col: [required] Target column.
        :param min_rep: [optional] minimum representativity value.

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-03", 35, 2, 2, 8], ["2021-03", 40, 4, 2, 7], ["2021-03", 50, 1, 1, 6], ["2021-03", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.TGrad(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5)
        >>> result_json = mine_obj.discover_tgp(parallel=True)
        >>> print(result_json)
        """

        super(TGrad, self).__init__(*args)
        self.target_col = target_col
        """:type: target_col: int"""
        self.min_rep = min_rep
        """:type: min_rep: float"""
        self.max_step = self.row_count - int(min_rep * self.row_count)
        """:type: max_step: int"""
        self.full_attr_data = self.data.copy().T
        """:type: full_attr_data: numpy.ndarray"""
        if len(self.time_cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            """:type: time_ok: bool"""
        else:
            print("Dataset Error")
            self.time_ok = False
            """:type: time_ok: bool"""
            raise Exception('No date-time datasets found')

    def discover_tgp(self, parallel: bool = False, num_cores: int = 1):
        """

        Applies fuzzy-logic, data transformation and gradual pattern mining to mine for Fuzzy Temporal Gradual Patterns.

        :param parallel: allow multiprocessing.
        :param num_cores: number of CPU cores for algorithm to use.
        :return: list of FTGPs as JSON object
        """

        self.gradual_patterns = []
        """:type: gradual_patterns: list(so4gp.TGP)"""
        str_gps = []

        # 1. Mine FTGPs
        if parallel:
            # implement parallel multi-processing
            steps = range(self.max_step)
            pool = mp.Pool(num_cores)
            patterns = pool.map(self.transform_and_mine, steps)
            pool.close()
            pool.join()
        else:
            patterns = list()
            for step in range(self.max_step):
                t_gps = self.transform_and_mine(step + 1)  # because for-loop is not inclusive from range: 0 - max_step
                if t_gps:
                    patterns.append(t_gps)

        # 2. Organize FTGPs into a single list
        for lst_obj in patterns:
            if lst_obj:
                for tgp in lst_obj:
                    self.gradual_patterns.append(tgp)
                    str_gps.append(tgp.print(self.titles))
        # Output
        out = json.dumps({"Algorithm": "TGrad", "Patterns": str_gps})
        """:type out: object"""
        return out

    def transform_and_mine(self, step: int, return_patterns: bool = True):
        """
        A method that: (1) transforms data according to a step value and, (2) mines the transformed data for FTGPs.

        :param step: data transformation step.
        :param return_patterns: allow method to mine TGPs.
        :return: list of TGPs
        """
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
                    # 2. Transform datasets
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

                    if return_patterns:
                        # 2. Execute t-graank for each transformation
                        t_gps = self.discover(time_delay_data=time_diffs, attr_data=delayed_attr_data)
                        if len(t_gps) > 0:
                            return t_gps
                        return False
                    else:
                        return delayed_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_time_diffs(self, step: int):  # optimized
        """

        A method that computes the difference between 2 timestamps separated by a specific transformation step.

        :param step: data transformation step.
        :return: set of time delay values
        """
        size = self.row_count
        time_diffs = {}  # {row: time-lag}
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
                # if time_diff < 0:
                # Error time CANNOT go backwards
                # print(f"Problem {i} and {i + step} - {self.time_cols}")
                #    return False, [i + 1, i + step + 1]
                time_diffs[int(i)] = float(abs(time_diff))
        return True, time_diffs

    def discover(self, time_delay_data: np.ndarray | dict = None, attr_data: np.ndarray = None,
                 clustering_method: bool = False):
        """

        Uses apriori algorithm to find GP candidates based on the target-attribute. The candidates are validated if
        their computed support is greater than or equal to the minimum support threshold specified by the user.

        :param time_delay_data: time-delay values
        :param attr_data: the transformed data.
        :param clustering_method: find and approximate best time-delay value using KMeans and Hill-climbing approach.
        :return: temporal-GPs as a list.
        """

        self.fit_bitmap(attr_data)

        gradual_patterns = []
        """:type gradual_patterns: list"""
        valid_bins = self.valid_bins

        if clustering_method:
            # Build the main triangular MF using clustering algorithm
            a, b, c = TGradAMI.build_mf_w_clusters(time_delay_data)
            tri_mf_data = np.array([a, b, c])
        else:
            tri_mf_data = None

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins, self.target_col)
            invalid_count += inv_count
            for v_bin in valid_bins:
                gi_arr = v_bin[0]
                bin_data = v_bin[1]
                sup = v_bin[2]
                gradual_patterns = TGP.remove_subsets(gradual_patterns, set(gi_arr))
                if type(self) is TGrad:
                    t_lag = self.get_fuzzy_time_lag(bin_data, time_delay_data, gi_arr=None, tri_mf_data=tri_mf_data)
                else:
                    t_lag = self.get_fuzzy_time_lag(bin_data, time_delay_data, gi_arr, tri_mf_data)

                if t_lag.valid:
                    tgp = TGP()
                    """:type gp: TGP"""
                    for obj in gi_arr:
                        gi = GI(obj[0], obj[1].decode())
                        """:type gi: GI"""
                        if gi.attribute_col == self.target_col:
                            tgp.add_target_gradual_item(gi)
                        else:
                            tgp.add_temporal_gradual_item(gi, t_lag)
                    tgp.set_support(sup)
                    gradual_patterns.append(tgp)
        return gradual_patterns

    def get_fuzzy_time_lag(self, bin_data: np.ndarray, time_data: np.ndarray | dict, gi_arr: set = None,
                           tri_mf_data: np.ndarray | None = None):
        """

        A method that uses fuzzy membership function to select the most accurate time-delay value. We implement two
        methods: (1) uses classical slide and re-calculate dynamic programming to find best time-delay value and,
        (2) uses metaheuristic hill-climbing to find the best time-delay value.

        :param bin_data: gradual item pairwise matrix.
        :param time_data: time-delay values.
        :param gi_arr: gradual item object.
        :param tri_mf_data: The a,b,c values of the triangular MF. Used to find and approximate best time-delay value
        using KMeans and Hill-climbing approach.
        :return: TimeDelay object.
        """

        # 1. Get Indices
        indices = np.argwhere(bin_data == 1)

        # 2. Get TimeDelay Array
        selected_rows = np.unique(indices.flatten())
        if gi_arr is not None:
            selected_cols = []
            for obj in gi_arr:
                # Ignore target-col and, remove time-cols and target-col from count
                col = int(obj[0])
                if (col != self.target_col) and (col < self.target_col):
                    selected_cols.append(col - (len(self.time_cols)))
                elif (col != self.target_col) and (col > self.target_col):
                    selected_cols.append(col - (len(self.time_cols) + 1))
            selected_cols = np.array(selected_cols, dtype=int)
            t_lag_arr = time_data[np.ix_(selected_cols, selected_rows)]
        else:
            time_lags = []
            for row, stamp_diff in time_data.items():  # {row: time-lag-stamp}
                if int(row) in selected_rows:
                    time_lags.append(stamp_diff)
            t_lag_arr = np.array(time_lags)
            best_time_lag = TGrad.approx_time_slide_calculate(t_lag_arr)
            return best_time_lag

        # 3. Approximate TimeDelay value
        best_time_lag = TimeDelay(-1, 0)
        """:type best_time_lag: so4gp.TimeDelay"""
        if tri_mf_data is not None:
            # 3b. Learn the best MF through KMeans and Hill-Climbing
            a, b, c = tri_mf_data
            best_time_lag = TimeDelay(-1, -1)
            fuzzy_set = []
            for t_lags in t_lag_arr:
                init_bias = abs(b - np.median(t_lags))
                slide_val, loss = TGradAMI.approx_time_hill_climbing(tri_mf_data, t_lags, initial_bias=init_bias)
                tstamp = int(b - slide_val)
                sup = float(1 - loss)
                fuzzy_set.append([tstamp, float(loss)])
                if sup >= best_time_lag.support and abs(tstamp) > abs(best_time_lag.timestamp):
                    best_time_lag = TimeDelay(tstamp, sup)
                # print(f"New Membership Fxn: {a - slide_val}, {b - slide_val}, {c - slide_val}")
        else:
            # 3a. Learn the best MF through slide-descent/sliding
            for t_lags in t_lag_arr:
                time_lag = TGrad.approx_time_slide_calculate(t_lags)
                if time_lag.support >= best_time_lag.support:
                    best_time_lag = time_lag
        return best_time_lag

    @staticmethod
    def get_timestamp(time_str: str):
        """

        A method that computes the corresponding timestamp from a DateTime string.

        :param time_str: DateTime value as a string
        :return: timestamp value
        """
        try:
            ok, stamp = DataGP.test_time(time_str)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False

    @staticmethod
    def triangular_mf(x: float, a: float, b: float, c: float):
        """

        A method that implements the fuzzy triangular membership function and computes the membership degree of value w.r.t
        the MF.

        :param x: value to be tested.
        :param a: left-side/minimum boundary of the triangular membership function.
        :param b: center value of the triangular membership function.
        :param c: maximum boundary value of the triangular membership function.
        :return: membership degree of value x.
        """
        if a <= x <= b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    @staticmethod
    def approx_time_slide_calculate(time_lags: np.ndarray):
        """

        A method that selects the most appropriate time-delay value from a list of possible values.

        :param time_lags: an array of all the possible time-delay values.
        :return: the approximated TimeDelay object.
        """

        if len(time_lags) <= 0:
            # if time_lags is blank return nothing
            return TimeDelay()
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
                    # print(boundaries[1])
                    return TimeDelay(int(boundaries[1]), sup)
            return TimeDelay(center, sup1)
