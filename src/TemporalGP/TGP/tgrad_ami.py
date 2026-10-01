# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

import time

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from so4gp import TGP

from .t_graank import TGrad


class TGradAMI(TGrad):
    def __init__(self, *args, **kwargs):
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
        :param kwargs: [optional] minimum representativity, [optional] MF shape, [optional] clustering algorithm,
        [optional] inference method.

        """
        super().__init__(*args, **kwargs)
        self._mi_error: float = 0
        self._transformation_data: dict = {}

    @property
    def mi_error(self):
        return self._mi_error

    @property
    def transformation_data(self):
        return self._transformation_data

    def get_mi_transformation_steps(
        self, error_margin: float
    ) -> tuple[dict[int, int], int]:
        """
        Estimate the optimal transformation step for each feature using
        Average Mutual Information (AMI).

        For each feature, this method computes the mutual information (MI)
        between the target attribute and:

        1. The original (untransformed) dataset.
        2. All candidate time-transformed datasets that satisfy the minimum
           representativity constraint.

        The optimal transformation is the one whose MI differs from the
        original dataset by at most the specified error margin. This approach
        assumes that the best transformation step that preserves the information shared
        between the feature and the target attribute.

        To simplify comparison during optimization, an MI value of zero
        (indicating no mutual information) is internally encoded as ``-1``.
        This sentinel value allows the algorithm to distinguish the absence
        of mutual information from very small positive MI values while
        preserving equality comparisons between the original and transformed
        datasets.

        Args:
            error_margin:
                Maximum allowable absolute difference between the mutual
                information of the original dataset and that of a transformed
                dataset.

        Returns:
            A dictionary mapping each feature column index to its selected
            transformation step (estimated time delay) and the maximum transformation step.

        Notes:
            Only transformed datasets satisfying the minimum representativity
            threshold are considered during the search for the optimal time
            transformation.
        """

        def return_results(steps_data, mi_err, maximum_step):
            self._mi_error = mi_err
            self.min_rep = round(((self.row_count - maximum_step) / self.row_count), 5)
            return steps_data, maximum_step

        # 1. Compute MI for original dataset w.r.t. target-col
        feature_cols = self.attr_cols
        y = np.array(self.full_attr_data[self.target_col], dtype=float).T
        x_data = np.array(self.full_attr_data[feature_cols], dtype=float).T
        init_mi_info = np.array(mutual_info_regression(x_data, y), dtype=float)

        # 2. Compute all the MI for every time-delay and compute error
        mi_list = []
        for step in range(1, self.max_step):
            # Compute MI
            steps_dict = {int(feature_cols[i]): step for i in range(len(feature_cols))}
            attr_data, _ = self.transform_data(steps_dict, step)
            if attr_data is None:
                return return_results(steps_dict, -1, step)

            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[feature_cols], dtype=float).T
            try:
                mi_vals = np.array(mutual_info_regression(x_data, y), dtype=float)
            except ValueError:
                return return_results(steps_dict, -1, step)

            # Compute MI error
            squared_diff = np.square(np.subtract(mi_vals, init_mi_info))
            mse_arr = np.sqrt(squared_diff)
            is_mi_preserved = np.all(mse_arr <= error_margin)
            if is_mi_preserved:
                return return_results(steps_dict, round(np.min(mse_arr), 5), step)
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
        steps_dict = {
            int(feature_cols[i]): int(optimal_steps_arr[i] + 1)
            for i in range(len(feature_cols))
        }
        return return_results(steps_dict, round(np.min(mse_arr), 5), max_step)

    def discover_tgp_ami(
        self,
        target_col: int,
        search_algorithm: str = "apriori",
        max_iteration: int|None = None,
        compute_descriptors: bool = False,
        transformation_steps: dict | None = None,
        ignore_time: bool = False,
        error_margin: float = 0.0001,
        eval_mode: bool = False,
    ) -> dict:
        """
        A method that applies mutual information concept, clustering, and hill-climbing algorithm to find the best data
        transformation that maintains MI and estimate the best time-delay value of the mined Fuzzy Temporal Gradual
        Patterns (FTGPs).

        :param target_col: [required] Index of the target attribute/feature/column. Temporal transformations are
        estimated relative to this attribute.
        :param search_algorithm: Gradual pattern mining algorithm to apply to the transformed dataset. Supported values
        are ``apriori``, ``ga``, ``aco``, ``pso``, ``hc``, ``random``, and ``clustergp``. Defaults to ``"apriori"``.
        :param max_iteration: Maximum number of iterations to run the search algorithm.
        :param compute_descriptors: If ``True``, compute descriptors for the mined gradual patterns. Defaults to ``False``.
        :param transformation_steps: Data transformation steps (used to override the computed transformation steps).
        :param ignore_time: Mine TGPs but skip the calculation and estimation of time delay.
        :param error_margin: [optional] minimum Mutual Information error margin.
        :param eval_mode: Run algorithm in evaluation mode.

        :return: List of (FTGPs as DICT object) or (FTGPs and evaluation data as a Python dict) when executed in evaluation mode.
        """

        start = time.time()
        self.target_col = target_col
        self._search_algorithm = search_algorithm
        self._algorithm_max_iter = max_iteration
        self._compute_descriptors = compute_descriptors
        self.clear_gradual_patterns()

        # 1. Compute and find the lowest mutual information (based) steps
        if transformation_steps is None:
            transformation_steps, max_step = self.get_mi_transformation_steps(
                error_margin=error_margin
            )
        else:
            max_step: int = 0
            for v in transformation_steps.values():
                max_step: int = max(max_step, v)

        # 3. Discover temporal-GPs from time-delayed data
        lst_tgp = self._safe_transform_and_mine(
            transformation_steps, max_step, skip_time=ignore_time
        )

        # 4. Organize FTGPs into a single list
        if lst_tgp:
            for tgp in lst_tgp:
                if isinstance(tgp, TGP):
                    self.add_gradual_pattern(tgp)

        # 5. Check if the algorithm is in evaluation mode
        if eval_mode:
            title_row = []
            time_title = []
            for col, txt in enumerate(self.titles):
                title_row.append(txt)
                if (col != self.target_col) and (col not in self.time_cols):
                    time_title.append(txt)
            transformed_data, time_data = self.transform_data(transformation_steps, max_step)
            str_time_data = {"".join(self.titles[k]): v for k, v in time_data.items()}
            self._transformation_data = {
                "Patterns": self.display_patterns,
                "Transformation Steps": transformation_steps,
                'Time Data': str_time_data,
                'Transformed Data': np.vstack(
                    (np.array(title_row), transformed_data.T if transformed_data is not None else np.array([]))),
            }

        duration = time.time() - start
        out_dict: dict[str, str | list | np.ndarray | None | dict] = {
            "Algorithm": "TGradAMI",
            # "Memory Usage (MiB)": f{mem_use)}",
            "GP Search Algorithm": f"{self._search_algorithm}",
            "Maximum Iteration for Search Algorithm": f"{self._algorithm_max_iter}",
            "Minimum Representation": f"{self.min_rep:.2f}",
            "MI Minimum Error": f"{error_margin:.2f}",
            "MI Error": f"{self.mi_error:.2f}",
            "Target Column": f"{self.target_col}",
            "Run-time": f"{duration:.6f} seconds",
        }
        out_dict.update({"Patterns": self.display_patterns})
        return out_dict
