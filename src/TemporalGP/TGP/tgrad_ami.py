# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

import time
import numpy as np
from sklearn.feature_selection import mutual_info_regression

from so4gp.algorithms import TGrad


class TGradAMI(TGrad):

    def __init__(self, *args, min_error: float = 0.0001, **kwargs):
        """
        Algorithm for estimating time-lag using Average Mutual Information (AMI) and KMeans clustering which is
        extended to mining gradual patterns. The average mutual information I(X; Y) is a measure of the “information”
        amount that the random variables X and Y provide about one another.

        This algorithm extends the work published in: https://ieeexplore.ieee.org/abstract/document/8858883. TGradAMI
        is an algorithm that improves the classical TGrad algorithm for extracting more accurate temporal gradual
        patterns.  It computes Mutual Information (MI) with respect to target-column with original dataset to get
        the actual relationship between variables: by computing MI for every possible time-delay and if the transformed
        dataset has the same almost identical MI to the original dataset, then it selects that as the best time-delay.
        Instead of min-representativity value, the algorithm relies on the error-margin between MIs.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param kwargs: [required] target-column or attribute or feature, [optional] minimum representativity
        :param min_error: [optional] minimum Mutual Information error margin.

        >>> from so4gp.algorithms import TGradAMI
        >>> import pandas
        >>>
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = TGradAMI(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5, min_error=0.1)
        >>> result_dict = mine_obj.discover_tgp(use_clustering=True, eval_mode=False)
        >>>
        >>> # print(result['Patterns'])
        >>> print(result_dict)
        """

        super(TGradAMI, self).__init__(*args, **kwargs)
        self._error_margin: float = min_error
        self._feature_cols: np.ndarray = np.setdiff1d(self.attr_cols, self.target_col)
        self._mi_error: float = 0

    @property
    def error_margin(self):
        return self._error_margin

    @property
    def mi_error(self):
        return self._mi_error

    @property
    def feature_cols(self):
        return self._feature_cols

    def find_best_mutual_info(self):
        """
        A method that computes the mutual information I(X; Y) of the original dataset and all the transformed datasets
        w.r.t. Minimum representativity threshold.

        We improve the computation of MI: if the MI of a dataset is 0 (0 indicates NO MI), we replace it with -1 (this
        encoding allows our algorithm to treat that MI as useless). So now, that allows our algorithm to easily
        distinguish very small MI values. This is beautiful because if the initial MI is 0, then both will be -1, making it
        the optimal MI with any other -1 in the time-delayed MIs.

        :return: Initial MI and MI for transformed datasets.
        """

        # 1. Compute MI for original dataset w.r.t. target-col
        y = np.array(self.full_attr_data[self.target_col], dtype=float).T
        x_data = np.array(self.full_attr_data[self._feature_cols], dtype=float).T
        init_mi_info = np.array(mutual_info_regression(x_data, y), dtype=float)

        # 2. Compute all the MI for every time-delay and compute error
        mi_list = []
        for step in range(1, self.max_step):
            # Compute MI
            attr_data, _ = self.transform_and_mine(step, return_patterns=False)
            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[self._feature_cols], dtype=float).T
            try:
                mi_vals = np.array(mutual_info_regression(x_data, y), dtype=float)
            except ValueError:
                optimal_dict = {int(self._feature_cols[i]): step for i in range(len(self._feature_cols))}
                self._mi_error = -1
                self.min_rep = round(((self.row_count - step) / self.row_count), 5)
                return optimal_dict, step

            # Compute MI error
            squared_diff = np.square(np.subtract(mi_vals, init_mi_info))
            mse_arr = np.sqrt(squared_diff)
            is_mi_preserved = np.all(mse_arr <= self._error_margin)
            if is_mi_preserved:
                optimal_dict = {int(self._feature_cols[i]): step for i in range(len(self._feature_cols))}
                self._mi_error = round(np.min(mse_arr), 5)
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
        max_step = int(np.max(optimal_steps_arr)) + 1

        # 5. Integrate feature indices with the computed steps
        optimal_dict = {int(self._feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self._feature_cols))}

        self._mi_error = round(np.min(mse_arr), 5)
        self.min_rep = round(((self.row_count - max_step) / self.row_count), 5)
        return optimal_dict, max_step

    def gather_delayed_data(self, optimal_dict: dict, max_step: int):
        """
        A method that combined attribute data with different data transformations and computes the corresponding
        time-delay values for each attribute.

        :param optimal_dict: Raw transformed dataset.
        :param max_step: Largest data transformation step.
        :return: Combined transformed dataset with corresponding time-delay values.
        """

        delayed_data: np.ndarray|None = None
        time_data = []
        n = self.row_count
        k = (n - max_step)  # Number of rows created by the largest step-delay
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
        return delayed_data, time_data

    def discover_tgp(self, use_clustering: bool = False, eval_mode: bool = False):
        """
        A method that applies mutual information concept, clustering, and hill-climbing algorithm to find the best data
        transformation that maintains MI and estimate the best time-delay value of the mined Fuzzy Temporal Gradual
        Patterns (FTGPs).

        :param use_clustering: Use a clustering algorithm to estimate the best time-delay value.
        :param eval_mode: Run algorithm in evaluation mode.
        :return: List of (FTGPs as DICT object) or (FTGPs and evaluation data as a Python dict) when executed in evaluation mode.
        """

        start = time.time()
        self.clear_gradual_patterns()
        # 1. Compute and find the lowest mutual information
        optimal_dict, max_step = self.find_best_mutual_info()

        # 2. Create a final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)

        # 3. Discover temporal-GPs from time-delayed data
        lst_tgp = self._mine_gps_at_step(time_delay_data=time_data, attr_data=delayed_data, clustering_method=use_clustering)

        # 4. Organize FTGPs into a single list
        if lst_tgp:
            for tgp in lst_tgp:
                self.add_gradual_pattern(tgp)

        # 5. Check if the algorithm is in evaluation mode
        if eval_mode:
            title_row = []
            time_title = []
            for col, txt in enumerate(self.titles):
                title_row.append(txt)
                if (col != self.target_col) and (col not in self.time_cols):
                    time_title.append(txt)
            add_dict = {
                'Patterns': self.display_patterns,
                'Time Data': np.vstack((np.array(time_title), time_data.T)),
                'Transformed Data': np.vstack((np.array(title_row), delayed_data.T if delayed_data is not None else np.array([]))),
            }
        else:
            add_dict = {"Patterns": self.display_patterns}

        duration = time.time() - start
        out_dict: dict[str, str | list | np.ndarray | None | dict] = {
            "Algorithm": "TGradAMI",
            # "Memory Usage (MiB)": f{mem_use)}",
            "Minimum Representation": f"{self.min_rep:.2f}",
            "MI Minimum Error": f"{self.error_margin:.2f}",
            "MI Error": f"{self.mi_error:.2f}",
            "Target Column": f"{self._target_col}",
            "Run-time": f"{duration:.6f} seconds"}
        self.generate_output_files(out_dict)

        out_dict.update(add_dict)
        return out_dict
