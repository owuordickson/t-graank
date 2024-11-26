# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Algorithm for estimating time-lag using AMI and extending it to mining gradual patterns.
The average mutual information I(X; Y) is a measure of the amount of “information” that
the random variables X and Y provide about one another.

"""

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

from so4gp import GI, TGP, GRAANK, TGrad


class TGradAMI(TGrad):
    """
    Algorithm for estimating time-lag using Average Mutual Information (AMI) and KMeans clustering which is extended to
    mining gradual patterns. The average mutual information I(X; Y) is a measure of the amount of “information” that
    the random variables X and Y provide about one another.

    This algorithm extends the work published in: https://ieeexplore.ieee.org/abstract/document/8858883.
    """

    def __init__(self, *args, min_error: float = 0.0001, **kwargs):
        """
        TGradAMI is an algorithm that improves the classical TGrad algorithm for extracting more accurate temporal
        gradual patterns. It computes Mutual Information (MI) with respect to target-column with original dataset to
        get the actual relationship between variables: by computing MI for every possible time-delay and if the
        transformed dataset has same almost identical MI to the original dataset, then it selects that as the best
        time-delay. Instead of min-representativity value, the algorithm relies on the error-margin between MIs.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param kwargs: [required] target-column or attribute or feature, [optional] minimum representativity
        :param min_error: [optional] minimum Mutual Information error margin.

        >>> import so4gp as sgp
        >>> import pandas
        >>> import json
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.TGradAMI(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5, min_error=0.1)
        >>> result_json = mine_obj.discover_tgp(use_clustering=True, eval_mode=False)
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json)
        """

        super(TGradAMI, self).__init__(*args, **kwargs)
        self.error_margin = min_error
        """:type error_margin: float"""
        self.feature_cols = np.setdiff1d(self.attr_cols, self.target_col)
        """:type feature_cols: numpy.ndarray"""
        self.mi_error = 0
        """:type mi_error: float"""

    def find_best_mutual_info(self):
        """
        A method that computes the mutual information I(X; Y) of the original dataset and all the transformed datasets
        w.r.t. minimum representativity threshold.

        We improve the computation of MI: if the MI of a dataset is 0 (0 indicates NO MI), we replace it with -1 (this
        encoding allows our algorithm to treat that MI as useless). So now, the allows our algorithm to easily
        distinguish very small MI values. This is beautiful because if initial MI is 0, then both will be -1 making it
        the optimal MI with any other -1 in the time-delayed MIs.

        :return: initial MI and MI for transformed datasets.
        """

        # 1. Compute MI for original dataset w.r.t. target-col
        y = np.array(self.full_attr_data[self.target_col], dtype=float).T
        x_data = np.array(self.full_attr_data[self.feature_cols], dtype=float).T
        init_mi_info = np.array(mutual_info_regression(x_data, y), dtype=float)

        # 2. Compute all the MI for every time-delay and compute error
        mi_list = []
        for step in range(1, self.max_step):
            # Compute MI
            attr_data, _ = self.transform_and_mine(step, return_patterns=False)
            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[self.feature_cols], dtype=float).T
            try:
                mi_vals = np.array(mutual_info_regression(x_data, y), dtype=float)
            except ValueError:
                optimal_dict = {int(self.feature_cols[i]): step for i in range(len(self.feature_cols))}
                """:type optimal_dict: dict"""
                self.mi_error = -1
                self.min_rep = round(((self.row_count - step) / self.row_count), 5)
                return optimal_dict, step

            # Compute MI error
            squared_diff = np.square(np.subtract(mi_vals, init_mi_info))
            mse_arr = np.sqrt(squared_diff)
            is_mi_preserved = np.all(mse_arr <= self.error_margin)
            if is_mi_preserved:
                optimal_dict = {int(self.feature_cols[i]): step for i in range(len(self.feature_cols))}
                """:type optimal_dict: dict"""
                self.mi_error = round(np.min(mse_arr), 5)
                self.min_rep = round(((self.row_count - step) / self.row_count), 5)
                return optimal_dict, step
            mi_list.append(mi_vals)
        mi_info_arr = np.array(mi_list, dtype=float)

        # 3. Standardize MI array
        mi_info_arr[mi_info_arr == 0] = -1

        # 4. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(mi_info_arr, init_mi_info))
        mse_arr = np.sqrt(squared_diff)
        # mse_arr[mse_arr < self.error_margin] = -1
        optimal_steps_arr = np.argmin(mse_arr, axis=0)
        max_step = int(np.max(optimal_steps_arr) + 1)
        """:type max_step: int"""

        # 5. Integrate feature indices with the computed steps
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        """:type optimal_dict: dict"""  # {col: steps}

        self.mi_error = round(np.min(mse_arr), 5)
        self.min_rep = round(((self.row_count - max_step) / self.row_count), 5)
        return optimal_dict, max_step

    def gather_delayed_data(self, optimal_dict: dict, max_step: int):
        """
        A method that combined attribute data with different data transformations and computes the corresponding
        time-delay values for each attribute.

        :param optimal_dict: raw transformed dataset.
        :param max_step: largest data transformation step.
        :return: combined transformed dataset with corresponding time-delay values.
        """

        delayed_data = None
        """:type delayed_data: numpy.ndarray | None"""
        time_data = []
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

        time_data = np.array(time_data)
        """:type time_data: numpy.ndarray"""
        return delayed_data, time_data

    def discover_tgp(self, use_clustering: bool = False, eval_mode: bool = False):
        """
        A method that applies mutual information concept, clustering and hill-climbing algorithm to find the best data
        transformation that maintains MI, and estimate the best time-delay value of the mined Fuzzy Temporal Gradual
        Patterns (FTGPs).

        :param use_clustering: use clustering algorithm to estimate the best time-delay value.
        :param eval_mode: run algorithm in evaluation mode.
        :return: list of (FTGPs as JSON object) or (FTGPs and evaluation data as a Python dict) when executed in evaluation mode.
        """

        self.gradual_patterns = []
        """:type: gradual_patterns: list(so4gp.TGP)"""
        str_gps = []

        # 1. Compute and find the lowest mutual information
        optimal_dict, max_step = self.find_best_mutual_info()

        # 2. Create final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)

        # 3. Discover temporal-GPs from time-delayed data
        if eval_mode:
            list_tgp, gp_components = self.extract_gradual_components(time_delay_data=time_data, attr_data=delayed_data,
                                                                   clustering_method=use_clustering)
            """:type t_gps: list"""
        else:
            list_tgp = self.__mine(time_delay_data=time_data, attr_data=delayed_data, clustering_method=use_clustering)
            """:type t_gps: list"""
            gp_components = None

        # 4. Organize FTGPs into a single list
        if list_tgp:
            for tgp in list_tgp:
                self.gradual_patterns.append(tgp)
                str_gps.append(tgp.print(self.titles))

        # 5. Check if algorithm is in evaluation mode
        if eval_mode:
            title_row = []
            time_title = []
            # print(eval_data)
            for txt in self.titles:
                col = int(txt[0])
                title_row.append(str(txt[1].decode()))
                if (col != self.target_col) and (col not in self.time_cols):
                    time_title.append(str(txt[1].decode()))
            eval_dict = {
                'Algorithm': 'TGradAMI',
                'Patterns': str_gps,
                'Time Data': np.vstack((np.array(title_row), delayed_data.T)),
                'Transformed Data': np.vstack((np.array(time_title), time_data.T)),
                'GP Components': gp_components
            }
            # Output
            return eval_dict
        else:
            # Output
            out = json.dumps({"Algorithm": "TGradAMI", "Patterns": str_gps})
            """:type out: object"""
            return out

    def extract_gradual_components(self, time_delay_data: np.ndarray | dict = None, attr_data: np.ndarray = None,
                                   clustering_method: bool = False):
        """
        A method that decomposes a multi-variate timeseries dataset into gradual components. The gradual components are
        warping paths represented as arrays. It also returns the mined fuzzy-temporal gradual patterns (FTGPs).

        :param time_delay_data: time-delay values as an array.
        :param attr_data: the transformed data.
        :param clustering_method: find and approximate best time-delay value using KMeans and Hill-climbing approach.
        :return: temporal-GPs as a list and gradual components as a Python dict object.
        """

        self.fit_bitmap(attr_data)
        valid_bins = self.valid_bins
        gradual_patterns = []
        """:type gradual_patterns: list"""
        gp_components = {}
        """:type gp_components: dict"""

        if clustering_method:
            # Build the main triangular MF using clustering algorithm
            a, b, c = TGradAMI.build_mf_w_clusters(time_delay_data)
            tri_mf_data = np.array([a, b, c])
        else:
            tri_mf_data = None

        for pairwise_obj in valid_bins:
            pairwise_mat = pairwise_obj[1]
            attr_col = pairwise_obj[0][0]
            attr_name = pairwise_obj[0][1].decode()
            gi = GI(attr_col, attr_name)
            gp_components[gi.to_string()] = GRAANK.decompose_to_gp_component(pairwise_mat)

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self.__gen_apriori_candidates(valid_bins, target_col=self.target_col)
            invalid_count += inv_count
            for v_bin in valid_bins:
                gi_arr = v_bin[0]
                bin_data = v_bin[1]
                sup = v_bin[2]
                gradual_patterns = TGP.remove_subsets(gradual_patterns, set(gi_arr))
                t_lag = self.get_fuzzy_time_lag(bin_data, time_delay_data, gi_arr, tri_mf_data)

                if t_lag.valid:
                    tgp = TGP()
                    for obj in gi_arr:
                        gi = GI(obj[0], obj[1].decode())
                        if gi.attribute_col == self.target_col:
                            tgp.add_target_gradual_item(gi)
                        else:
                            tgp.add_temporal_gradual_item(gi, t_lag)
                    tgp.set_support(sup)
                    gradual_patterns.append(tgp)
                    gp_components[f"{tgp.to_string()}"] = GRAANK.decompose_to_gp_component(bin_data)

        return gradual_patterns, gp_components

    @staticmethod
    def build_mf_w_clusters(time_data: np.ndarray):
        """
        A method that builds the boundaries of a fuzzy Triangular membership function (MF) using Singular Value
        Decomposition (to estimate the number of centers) and KMeans algorithm to group time data according to the
        identified centers. We then use the largest cluster to build the MF.

        :param time_data: time-delay values as an array.
        :return: the a, b, c boundary values of the triangular membership function.
        """

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
            half_width = 0.5 / 2  # since membership value should be > 0.5
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

    @staticmethod
    def approx_time_hill_climbing(tri_mf: np.ndarray, x_train: np.ndarray, initial_bias: float = 0,
                                  step_size: float = 0.9, max_iterations: int = 10):
        """
        A method that uses Hill-climbing algorithm to approximate the best time-delay value given a fuzzy triangular
        membership function.

        :param tri_mf: fuzzy triangular membership function boundaries (a, b, c) as an array.
        :param x_train: initial time-delay values as an array.
        :param initial_bias: (hyperparameter) initial bias value for the hill-climbing algorithm.
        :param step_size: (hyperparameter) step size for the hill-climbing algorithm.
        :param max_iterations: (hyperparameter) maximum number of iterations for the hill-climbing algorithm.
        :return: best position to move the triangular MF with its mean-squared-error.
        """

        # 1. Normalize x_train
        x_train = np.array(x_train, dtype=float)

        # 2. Perform hill climbing to find the optimal bias
        bias = initial_bias
        y_train = x_train + bias
        best_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf)
        for iteration in range(max_iterations):
            # a. Generate a new candidate bias by perturbing the current bias
            new_bias = bias + step_size * np.random.randn()

            # b. Compute the predictions and the MSE with the new bias
            y_train = x_train + new_bias
            new_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf)

            # c. If the new MSE is lower, update the bias
            if new_mse < best_mse:
                bias = new_bias
                best_mse = new_mse

        # Make predictions using the optimal bias
        return bias, best_mse

    @staticmethod
    def hill_climbing_cost_function(y_train: np.ndarray, tri_mf: np.ndarray, min_membership: float = 0):
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
        memberships = np.where(y_train <= b,
                               (y_train - a) / (b - a),
                               (c - y_train) / (c - b))

        # 2. Generate y_train based on the given criteria (x>minimum_membership)
        y_hat = np.where(memberships >= min_membership, 1, 0)

        # 3. Compute loss
        hat_count = np.count_nonzero(y_hat)
        true_count = len(y_hat)
        loss = (((true_count - hat_count) / true_count) ** 2) ** 0.5
        """:type loss: float"""
        # loss = abs(true_count - hat_count)
        return loss
