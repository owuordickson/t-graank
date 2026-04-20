# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Algorithm for mining temporal gradual patterns using fuzzy membership functions.
"""


import json
import time
import numpy as np
import skfuzzy as fuzzy
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from so4gp import DataGP, GI, TGP, TimeDelay
from so4gp.algorithms import GRAANK


class TGrad(GRAANK):

    def __init__(self, *args, target_col: int, min_rep: float = 0.5, **kwargs):
        """
        TGrad is an algorithm used to extract temporal gradual patterns from numeric datasets. An algorithm for mining
        temporal gradual patterns using fuzzy membership functions. It uses a technique
        published in: https://ieeexplore.ieee.org/abstract/document/8858883.

        :param args: [required] a data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param target_col: [required] Target column.
        :param min_rep: [optional] minimum representativity value.

        >>> import so4gp.algorithms import TGrad
        >>> import pandas
        >>>
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = TGrad(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5)
        >>> result_json = mine_obj.discover_tgp(parallel=True)
        >>> # print(result['Patterns'])
        >>> print(result_json)
        """

        super(TGrad, self).__init__(*args, **kwargs)
        self._target_col: int = target_col
        self._min_rep: float = min_rep
        self._max_step: int = self.row_count - int(min_rep * self.row_count)
        self._full_attr_data: np.ndarray = self.data.copy().T
        if len(self.time_cols) > 0:
            print("Dataset Ok")
            self._time_ok: bool = True
        else:
            print("Dataset Error")
            self._time_ok: bool = False
            raise Exception('No date-time datasets found')

    @property
    def target_col(self):
        return self._target_col

    @property
    def min_rep(self):
        return self._min_rep

    @property
    def max_step(self):
        return self._max_step

    @property
    def full_attr_data(self):
        return self._full_attr_data

    @min_rep.setter
    def min_rep(self, value):
        if 0 < value <= 1:
            self._min_rep = value

    def discover_tgp(self, parallel: bool = False, num_cores: int = 1):
        """
        Applies fuzzy-logic, data transformation, and gradual pattern mining to mine for Fuzzy Temporal Gradual Patterns.

        :param parallel: Allow multiprocessing.
        :param num_cores: Number of CPU cores for the algorithm to use.
        :return: List of FTGPs as JSON object
        """

        start = time.time()
        self.clear_gradual_patterns()
        # 1. Mine FTGPs
        if parallel:
            # implement parallel multi-processing
            with mp.Pool(num_cores) as pool:
                steps = range(self._max_step)
                pattern_data = pool.map(self._safe_transform_and_mine, steps)
        else:
            pattern_data: list = []
            for step in range(self._max_step):
                t_gps = self._safe_transform_and_mine(step + 1)  # because for-loop it is not inclusive from range: 0 - max_step
                pattern_data.append(t_gps)

        # 2. Organize FTGPs into a single list
        for item in pattern_data:
            if item is None:
                continue

            # Standardize 'item' into a list so we only need one loop
            lst_pattern = item if isinstance(item, list) else [item]

            for pat in lst_pattern:
                if isinstance(pat, TGP):
                    self.add_gradual_pattern(pat)

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "TGrad",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Run-time": f"{duration:.6f} seconds"}
        self.generate_output_files(out_dict)

        out_dict.update({"Patterns": self.display_patterns})
        out: object = json.dumps(out_dict, indent=4)
        return out

    def transform_and_mine(self, step: int, return_patterns: bool = True):
        """
        A method that: (1) transforms data according to a step value and, (2) mines the transformed data for FTGPs.

        :param step: Data transformation step.
        :param return_patterns: Allow method to mine TGPs.
        :return: List of TGPs
        """
        # NB: Restructure dataset based on target/reference col
        if self._time_ok:
            # 1. Calculate the time difference using a step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                tgt_col = self._target_col
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
                            temp_row = self._full_attr_data[col_index][0: (n - step)]
                        else:
                            # other attributes
                            temp_row = self._full_attr_data[col_index][step: n]

                        delayed_attr_data = temp_row if (delayed_attr_data is None) \
                            else np.vstack((delayed_attr_data, temp_row))
                    # print(f"Time Diffs: {time_diffs}\n")
                    # print(f"{self.full_attr_data}: {type(self.full_attr_data)}\n")
                    # print(f"{delayed_attr_data}: {type(delayed_attr_data)}\n")

                    if return_patterns:
                        # 2. Execute t-graank for each transformation
                        t_gps = self._mine_gps_at_step(time_delay_data=time_diffs, attr_data=delayed_attr_data)
                        if len(t_gps) > 0:
                            return t_gps
                        return False
                    else:
                        return delayed_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def _safe_transform_and_mine(self, step: int, return_patterns: bool = True):
        """Wrapper to catch exceptions during parallel mining."""
        try:
            return self.transform_and_mine(step, return_patterns=return_patterns)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            return None

    def _mine_gps_at_step(self, time_delay_data: np.ndarray | dict, attr_data: np.ndarray = None, clustering_method: bool = False) -> list[TGP] | tuple[list[TGP], dict]:
        """
        Uses apriori algorithm to find GP candidates based on the target-attribute. The candidates are validated if
        their computed support is greater than or equal to the minimum support threshold specified by the user.

        :param time_delay_data: Time-delay values
        :param attr_data: the transformed data.
        :param clustering_method: Find and approximate the best time-delay value using KMeans and Hill-climbing approach.
        :return: Temporal-GPs as a list.
        """

        try:
            # If min-rep is too low
            self.fit_bitmap(attr_data)
        except ZeroDivisionError:
            return []

        t_gps: list[TGP] = []
        valid_bins_dict: dict = (self.valid_bins or {}).copy()

        if clustering_method and isinstance(time_delay_data, np.ndarray):
            # Build the main triangular MF using the clustering algorithm
            a, b, c = TGrad.build_mf_w_clusters(time_delay_data)
            tri_mf_data = np.array([a, b, c])
        else:
            tri_mf_data = None

        invalid_count = 0
        while len(valid_bins_dict) > 0:
            valid_bins_dict, inv_count = self._gen_apriori_candidates(valid_bins_dict, target_col=self._target_col)
            invalid_count += inv_count
            for gp_set, gi_data in valid_bins_dict.items():
                if type(self) is TGrad:
                    t_lag = self.get_fuzzy_time_lag(gi_data.bin_mat, time_delay_data, gi_arr=None, tri_mf_data=tri_mf_data)  # dict
                else:
                    t_lag = self.get_fuzzy_time_lag(gi_data.bin_mat, time_delay_data, gi_arr=gp_set, tri_mf_data=tri_mf_data)  # array

                if t_lag.valid:
                    tgp: TGP = TGP()
                    for gi_str in gp_set:
                        gi: GI = GI.from_string(gi_str)
                        if gi.attribute_col == self._target_col:
                            tgp.target_gradual_item = gi
                        else:
                            tgp.add_temporal_gradual_item(gi, t_lag)
                    tgp.support = gi_data.support
                    warping_set_arr: np.ndarray = np.array(
                    DataGP.gen_gradual_warping_set(gi_data.bin_mat, as_array=True))
                    tgp.compute_descriptors(warping_set_arr, obj_count=self.row_count)
                    t_gps.append(tgp)
        return t_gps

    def get_time_diffs(self, step: int):  # optimized
        """
        A method that computes the difference between 2 timestamps separated by a specific transformation step.

        :param step: Data transformation step.
        :return: Dict of time delay values
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
                # Error time CANNOT go backwards,
                # print(f"Problem {i} and {i + step} - {self.time_cols}")
                #    return False, [i + 1, i + step + 1]
                time_diffs[int(i)] = float(abs(time_diff))
        return True, time_diffs

    def get_fuzzy_time_lag(self, bin_data: np.ndarray, time_data: np.ndarray | dict, gi_arr: set = None, tri_mf_data: np.ndarray | None = None) -> TimeDelay:
        """
        A method that uses a fuzzy membership function to select the most accurate time-delay value. We implement two
        methods: (1) uses classical slide and re-calculate dynamic programming to find the best time-delay value and,
        (2) uses metaheuristic hill-climbing to find the best time-delay value.

        :param bin_data: Gradual item pairwise matrix.
        :param time_data: Time-delay values.
        :param gi_arr: Gradual item object.
        :param tri_mf_data: The 'a,b,c' values of the triangular MF. Used to find and approximate the best time-delay value
        using KMeans and Hill-climbing approach.

        :return: TimeDelay object.
        """

        if time_data is None:
            return TimeDelay()

        time_data_as_arr: np.ndarray|None = time_data if isinstance(time_data, np.ndarray)else None
        time_data_as_dict: dict|None = time_data if isinstance(time_data, dict) else None

        def approx_time_slide_calculate(time_lag_arr: np.ndarray) -> TimeDelay:
            """

            A method that selects the most appropriate time-delay value from a list of possible values.

            :param time_lag_arr: An array of all the possible time-delay values.
            :return: The approximated TimeDelay object.
            """

            if len(time_lag_arr) <= 0:
                # if time_lags is blank, return nothing
                return TimeDelay()
            else:
                time_lag_arr = np.absolute(np.array(time_lag_arr))
                min_a = np.min(time_lag_arr)
                max_c = np.max(time_lag_arr)
                count = time_lag_arr.size + 3
                tot_boundaries = np.linspace(min_a / 2, max_c + 1, num=count)

                highest_sup = 0
                center = time_lag_arr[0]
                size = len(tot_boundaries)
                for i in range(0, size, 2):
                    if (i + 3) <= size:
                        boundaries = tot_boundaries[i:i + 3:1]
                    else:
                        boundaries = tot_boundaries[size - 3:size:1]
                    memberships = fuzzy.membership.trimf(time_lag_arr, boundaries)

                    # Compute Support
                    sup_count = np.count_nonzero(memberships > 0)
                    total = memberships.size
                    curr_sup = sup_count / total
                    # curr_sup = calculate_support(memberships)

                    if curr_sup > highest_sup:
                        highest_sup = curr_sup
                        center = boundaries[1]
                    if curr_sup >= 0.5:
                        # print(boundaries[1])
                        return TimeDelay(int(boundaries[1]), curr_sup)
                return TimeDelay(center, highest_sup)

        def approx_time_hill_climbing(x_train: np.ndarray, initial_bias: float = 0, step_size: float = 0.9, max_iterations: int = 10):
            """
            A method that uses Hill-climbing algorithm to approximate the best time-delay value given a fuzzy triangular
            membership function.

            :param x_train: Initial time-delay values as an array.
            :param initial_bias: (hyperparameter) initial bias value for the hill-climbing algorithm.
            :param step_size: (hyperparameter) step size for the hill-climbing algorithm.
            :param max_iterations: (hyperparameter) maximum number of iterations for the hill-climbing algorithm.
            :return: Best position to move the triangular MF with its mean-squared-error.
            """

            def hill_climbing_cost_function(min_membership: float = 0):
                """
                Computes the logistic regression cost function for a fuzzy set created from a
                triangular membership function.

                :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
                :return: Cost function values.
                """
                # 1. Generate fuzzy data set using MF from x_data
                memberships = np.where(y_train <= b,
                                       (y_train - a) / (b - a),
                                       (c - y_train) / (c - b))

                # 2. Generate y_train based on the given criteria (x>minimum_membership)
                y_hat = np.where(memberships >= min_membership, 1, 0)

                # 3. Compute loss_val
                hat_count = np.count_nonzero(y_hat)
                true_count = len(y_hat)
                loss_val: float = (((true_count - hat_count) / true_count) ** 2) ** 0.5
                # loss_val = abs(true_count - hat_count)
                return loss_val

            # 1. Normalize x_train
            x_train = np.array(x_train, dtype=float)

            # 2. Perform hill climbing to find the optimal bias
            bias = initial_bias
            y_train = x_train + bias
            best_mse = hill_climbing_cost_function()
            for iteration in range(max_iterations):
                # a. Generate a new candidate bias by perturbing the current bias
                new_bias = bias + step_size * np.random.randn()

                # b. Compute the predictions and the MSE with the new bias
                y_train = x_train + new_bias
                new_mse = hill_climbing_cost_function()

                # c. If the new MSE is lower, update the bias
                if new_mse < best_mse:
                    bias = new_bias
                    best_mse = new_mse

            # Make predictions using the optimal bias
            return bias, best_mse

        # 1. Get Indices
        indices = np.argwhere(bin_data == 1)

        # 2. Get TimeDelay Array
        selected_rows = np.unique(indices.flatten())
        if gi_arr is not None:
            selected_cols = []
            for gi_str in gi_arr:
                # Ignore target-col and remove time-cols and target-col from the count
                col = GI.from_string(gi_str).attribute_col
                if (col != self._target_col) and (col < self._target_col):
                    selected_cols.append(col - (len(self.time_cols)))
                elif (col != self._target_col) and (col > self._target_col):
                    selected_cols.append(col - (len(self.time_cols) + 1))
            selected_cols = np.array(selected_cols, dtype=int)
            t_lag_arr = time_data_as_arr[np.ix_(selected_cols, selected_rows)] if time_data_as_arr is not None else np.array([])
        else:
            time_lags = []
            for row, stamp_diff in (time_data_as_dict or {}).items():  # {row: time-lag-stamp}
                if int(row) in selected_rows:
                    time_lags.append(stamp_diff)
            t_lag_arr = np.array(time_lags)
            best_time_lag = approx_time_slide_calculate(t_lag_arr)
            return best_time_lag

        # 3. Approximate TimeDelay value
        best_time_lag: TimeDelay = TimeDelay(-1, 0)
        if tri_mf_data is not None:
            # 3b. Learn the best MF through KMeans and Hill-Climbing
            a, b, c = tri_mf_data
            best_time_lag = TimeDelay(-1, -1)
            fuzzy_set = []
            for t_lags in t_lag_arr:
                init_bias = abs(b - np.median(t_lags))
                slide_val, loss = approx_time_hill_climbing(t_lags, initial_bias=init_bias)
                tstamp = int(b - slide_val)
                sup = float(1 - loss)
                fuzzy_set.append([tstamp, float(loss)])
                if sup >= best_time_lag.support and abs(tstamp) > abs(best_time_lag.timestamp):
                    best_time_lag = TimeDelay(tstamp, sup)
                # print(f"New Membership Fxn: {a - slide_val}, {b - slide_val}, {c - slide_val}")
        else:
            # 3a. Learn the best MF through slide-descent/sliding
            for t_lags in t_lag_arr:
                time_lag = approx_time_slide_calculate(t_lags)
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
    def build_mf_w_clusters(time_data: np.ndarray|None):
        """
        A method that builds the boundaries of a fuzzy Triangular membership function (MF) using Singular Value
        Decomposition (to estimate the number of centers) and KMeans algorithm to group time data according to the
        identified centers. We then use the largest cluster to build the MF.

        :param time_data: Time-delay values as an array.
        :return: The boundary values of the triangular membership function.
        """

        if time_data is None:
            return 0, 0, 0

        try:
            # 1. Reshape into 1-column dataset
            time_data = time_data.reshape(-1, 1)

            # 2. Standardize data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(time_data)

            # 3. Apply SVD
            u, s, vt = np.linalg.svd(data_scaled, full_matrices=False)

            # 4. Plot singular values to help determine the number of clusters
            # Based on the plot, choose the number of clusters (e.g., 3 clusters)
            num_clusters = int(s[0])

            # 5. Perform k-means clustering
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data_scaled)

            # 6. Get cluster centers
            centers = kmeans.cluster_centers_.flatten()

            # 7. Define membership functions to ensure membership > 0.5
            largest_mf = [0, 0, 0]
            for center in centers:
                half_width = 0.5 / 2  # since the membership value should be > 0.5
                a = center - half_width
                b = center
                c = center + half_width
                if abs(c - a) > abs(largest_mf[2] - largest_mf[0]):
                    largest_mf = [a, b, c]

            # 8. Reverse the scaling
            a = scaler.inverse_transform([[largest_mf[0]]])[0, 0]
            b = scaler.inverse_transform([[largest_mf[1]]])[0, 0]
            c = scaler.inverse_transform([[largest_mf[2]]])[0, 0]

            # 9. Shift to remove negative MF (we do not want negative timestamps)
            if a < 0:
                shift_by = abs(a)
                a = a + shift_by
                b = b + shift_by
                c = c + shift_by
            return a, b, c
        except Exception as e:
            print(e)
            return 0, 0, 0
