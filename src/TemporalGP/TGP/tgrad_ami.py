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
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

from so4gp import TimeDelay
from .t_graank import TGrad


class TGradAMI(TGrad):

    def __init__(self, f_path: str, eq: bool, min_sup: float, target_col: int, err: float, num_cores: int):
        """"""
        # Compute MI w.r.t. target-column with original dataset to get the actual relationship
        # between variables. Compute MI for every time-delay/time-lag: if the values are
        # almost equal to actual, then we have the most accurate time-delay. Instead of
        # min-representativity value, we propose error-margin.

        super(TGradAMI, self).__init__(f_path, eq, min_sup, target_col=target_col, min_rep=0.25, num_cores=num_cores)
        self.error_margin = err
        self.min_membership = 0.001
        self.tri_mf_data = None  # The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
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

    def gather_delayed_data(self, optimal_dict: dict, max_step: int):
        """"""
        delayed_data = None
        time_data = []
        # time_dict = {}
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

                # Get first k items for delayed data
                temp_row = temp_row[0: k]

                # Get first k items for time-lag data
                temp_diffs = [(time_diffs[i]) for i in range(k)]
                time_data.append(temp_diffs)

                # for i in range(k):
                #    if i in time_dict:
                #        time_dict[i].append(time_diffs[i])
                #    else:
                #        time_dict[i] = [time_diffs[i]]
                # print(f"{time_diffs}\n")
                # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
            delayed_data = temp_row if (delayed_data is None) \
                else np.vstack((delayed_data, temp_row))

        # print(f"{time_dict}\n")
        return delayed_data, np.array(time_data)

    def discover_tgp(self, parallel=False, eval_mode=False):
        """"""

        # 1. Compute mutual information
        self.compute_mutual_info()

        # 2. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(self.mi_arr, self.initial_mutual_info))
        absolute_error = np.sqrt(squared_diff)
        optimal_steps_arr = np.argmin(absolute_error, axis=0)
        max_step = (np.max(optimal_steps_arr) + 1)
        # print(f"Largest step delay: {max_step}\n")
        # print(f"Abs.E.: {absolute_error}\n")

        # 3. Integrate feature indices with the computed steps
        # optimal_dict = dict(map(lambda key, val: (int(key), int(val+1)), self.feature_cols, optimal_steps_arr))
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        # print(f"Optimal Dict: {optimal_dict}\n")  # {col: steps}

        # 4. Create final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)
        # print(f"{delayed_data}\n")
        # print(f"Time Lags: {time_data}\n")

        # 5. Build triangular MF
        a, b, c = TGradAMI.build_mf(time_data)
        self.tri_mf_data = np.array([a, b, c])
        # print(f"Membership Function: {a}, {b}, {c}\n")

        # 6. Discover temporal-GPs from time-delayed data
        # 6a. Learn the best MF through slide-descent/sliding
        # 6b. Apply cartesian product on multiple MFs to pick the MF with the best center (inference logic)
        # Mine tGPs and then compute Union of time-lag MFs,
        # from this union select the MF with more members (little loss)
        t_gps = self.discover(t_diffs=time_data, attr_data=delayed_data)

        if len(t_gps) > 0:
            if eval_mode:
                title_row = []
                time_title = []
                for txt in self.titles:
                    col = int(txt[0])
                    title_row.append(str(txt[1].decode()))
                    if (col != self.target_col) and (col not in self.time_cols):
                        time_title.append(str(txt[1].decode()))

                return t_gps, np.vstack((np.array(title_row), delayed_data.T)), np.vstack((np.array(time_title), time_data.T))
            else:
                return t_gps
        return False

    def get_fuzzy_time_lag(self, bin_data: np.ndarray, time_diffs, gi_arr=None):
        """"""

        # 1. Get Indices
        indices = np.argwhere(bin_data == 1)

        # 2. Get TimeDelay Array
        selected_rows = np.unique(indices.flatten())
        selected_cols = []
        for obj in gi_arr:
            # Ignore target-col and, remove time-cols and target-col from count
            col = int(obj[0])
            if (col != self.target_col) and (col < self.target_col):
                selected_cols.append(col - (len(self.time_cols)))
            elif (col != self.target_col) and (col > self.target_col):
                selected_cols.append(col - (len(self.time_cols)+1))
        selected_cols = np.array(selected_cols, dtype=int)
        t_lag_arr = time_diffs[np.ix_(selected_cols, selected_rows)]

        # 3. Learn the best MF through slide-descent/sliding
        a, b, c = self.tri_mf_data
        best_time_lag = TimeDelay(-1, 0)
        fuzzy_set = []
        for t_lags in t_lag_arr:
            init_bias = abs(b-np.median(t_lags))
            slide_val, loss = TGradAMI.select_mf_hill_climbing(a, b, c, t_lags, initial_bias=init_bias)
            tstamp = int(b - slide_val)
            sup = float(1-loss)
            fuzzy_set.append([tstamp, float(loss)])
            if sup >= best_time_lag.support and tstamp > best_time_lag.timestamp:
                best_time_lag = TimeDelay(tstamp, sup)
            # print(f"New Membership Fxn: {a - slide_val}, {b - slide_val}, {c - slide_val}")

        # 4. Apply cartesian product on multiple MFs to pick the MF with the best center (inference logic)
        # Mine tGPs and then compute Union of time-lag MFs,
        # from this union select the MF with more members (little loss)

        # print(f"indices {selected_cols}: {selected_rows}")
        # print(f"GIs: {gi_arr}")
        # print(f"Fuzzy Set: {fuzzy_set}")
        # print(f"Selected Time Lag: {best_time_lag.to_string()}")
        # print(f"time lags: {t_lag_arr}")
        # print("\n")

        return best_time_lag

    @staticmethod
    def build_mf(time_data: np.ndarray):
        """"""

        # Reshape into 1-column dataset
        time_data = time_data.reshape(-1, 1)

        # Standardize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(time_data)

        # Apply SVD
        u, s, vt = np.linalg.svd(data_scaled, full_matrices=False)

        # Plot singular values to help determine the number of clusters
        # Based on the plot, choose the number of clusters (e.g., 3 clusters)
        num_clusters = int(s[0])

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data_scaled)

        # Get cluster centers
        centers = kmeans.cluster_centers_.flatten()

        # Define membership functions to ensure membership > 0.5
        # mf_list = []
        largest_mf = [0, 0, 0]
        for center in centers:
            half_width = 0.5 / 2  # since membership value should be > 0.5
            a = center - half_width
            b = center
            c = center + half_width
            if abs(c - a) > abs(largest_mf[2] - largest_mf[0]):
                largest_mf = [a, b, c]
            # mf_list.append((a, b, c))

        # Reverse the scaling
        a = scaler.inverse_transform([[largest_mf[0]]])[0, 0]
        b = scaler.inverse_transform([[largest_mf[1]]])[0, 0]
        c = scaler.inverse_transform([[largest_mf[2]]])[0, 0]

        # Shift to remove negative MF (we do not want negative timestamps)
        if a < 0:
            shift_by = abs(a)
            a = a + shift_by
            b = b + shift_by
            c = c + shift_by
        return a, b, c

    @staticmethod
    def select_mf_hill_climbing(a: float, b: float, c: float, x_train: np.ndarray,
                                initial_bias: float = 0, step_size: float = 0.9, max_iterations: int = 10):
        """"""

        # Normalize x_train
        x_train = np.array(x_train, dtype=float)
        # print(f"x-train: {x_train}")

        # Perform hill climbing to find the optimal bias
        min_membership = 0.001
        bias = initial_bias
        y_train = x_train + bias
        tri_mf = np.array([a, b, c])
        best_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf, min_membership)
        for iteration in range(max_iterations):
            # Generate a new candidate bias by perturbing the current bias
            new_bias = bias + step_size * np.random.randn()

            # Compute the predictions and the MSE with the new bias
            y_train = x_train + new_bias
            new_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf, min_membership)

            # If the new MSE is lower, update the bias
            if new_mse < best_mse:
                # print(f"new bias: {new_bias}")
                bias = new_bias
                best_mse = new_mse

        # Make predictions using the optimal bias
        y_train = x_train + bias
        # print(f"Optimal bias: {bias}")
        # print(f"Predictions: {y_train}")
        # print(f"Mean Squared Error: {best_mse*100}%")
        return bias, best_mse

    @staticmethod
    def hill_climbing_cost_function(y_train: np.ndarray, tri_mf: np.ndarray, min_membership: float = 0.5):
        """
        Computes the logistic regression cost function for a fuzzy set created from a
        triangular membership function.

        :param y_train: A numpy array of the predicted labels.
        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
        :return: cost function values.
        """

        a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]

        # 1. Generate fuzzy data set using MF from x_data
        y_hat = np.where(y_train <= b,
                         (y_train - a) / (b - a),
                         (c - y_train) / (c - b))
        # 2. Generate y_train based on the given criteria (x>minimum_membership)
        y_hat = np.where(y_hat >= min_membership, 1, 0)

        # 3. Compute loss
        hat_count = np.count_nonzero(y_hat)
        true_count = len(y_hat)
        loss = (((true_count - hat_count)/true_count) ** 2) ** 0.5
        # loss = abs(true_count - hat_count)
        return loss
