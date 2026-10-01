# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Algorithm for mining temporal gradual patterns using fuzzy membership functions.
"""


import copy
import time
import numpy as np
import pandas as pd

from so4gp import DataGP, TGP
from so4gp.algorithms import GRAANK
from so4gp.gradual_patterns import NO_TIME_LABEL, FatalError
from so4gp.algorithms.base.graank_alg import OrigGRAANK


class TGrad(OrigGRAANK):
    def __init__(
        self,
        *args,
        min_rep: float = 0.5,
        mf_shape: str = "gaussian",
        clustering_algorithm: str = "kmeans",
        inference_method: str = "mamdani",
        **kwargs,
    ):
        """
        TGrad is an algorithm used to extract temporal gradual patterns from numeric datasets. An algorithm for mining
        temporal gradual patterns using fuzzy membership functions. It uses a technique
        published in: https://ieeexplore.ieee.org/abstract/document/8858883.

        :param args: [required] a data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param min_rep: [optional] minimum representativity value.
        :param mf_shape: [optional] shape of the fuzzy membership function. Options are: 'triangular', 'trapezoidal', 'gaussian'.
        :param clustering_algorithm: [optional] clustering algorithm for estimating the MFs. Options are: 'kmeans', 'fcm'.
        :param inference_method: [optional] inference method for estimating the MFs. Options: 'mamdani', larsen'.

        """
        super().__init__(*args, **kwargs)
        self._search_algorithm: str = "apriori"
        self._algorithm_max_iter: int|None = None
        self._compute_descriptors: bool = False
        self._min_rep: float = min_rep
        self._max_step: int = self.row_count - int(min_rep * self.row_count)
        self.mf_shape = mf_shape.lower()
        self.clustering_algorithm = clustering_algorithm.lower()
        self.inference_method = inference_method.lower()
        self._full_attr_data: np.ndarray = copy.deepcopy(self.data).T
        if len(self.time_cols) > 0:
            # print("Dataset Ok")
            self._time_ok: bool = True
        else:
            # print("Dataset Error")
            self._time_ok: bool = False
            raise FatalError("No date-time datasets found")

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

    def discover_tgp(
        self,
        target_col: int,
        search_algorithm: str = "apriori",
        max_iteration: int|None = None,
        compute_descriptors: bool = False,
        ignore_time: bool = False,
    ) -> dict:
        """
        Mine Fuzzy Temporal Gradual Patterns (FTGPs) from a temporal dataset.

        The method applies the complete temporal gradual pattern mining pipeline:
        fuzzy-logic-based data transformation, temporal transformation, and
        gradual pattern mining. The temporal transformations are estimated
        relative to a specified target attribute, after which a gradual pattern
        search algorithm is applied to the transformed dataset.

        The gradual pattern search can use the classical APRIORI (GRAANK) algorithm or
        one of several metaheuristic and alternative search strategies.

        Supported search algorithms are:

        * ``apriori``:
          Classical APRIORI level-wise search for exhaustive gradual pattern
          candidate generation.

        * ``ga``:
          Genetic GRAANK. Uses a genetic algorithm to search the gradual pattern
          space.

        * ``aco``:
          ACO-GRAANK. Uses Ant Colony Optimization and pheromone-guided search
          to identify promising gradual pattern candidates.

        * ``pso``:
          PSO-GRAANK. Uses Particle Swarm Optimization to search for high-support
          gradual patterns.

        * ``hc``:
          Hill-Climbing GRAANK. Iteratively searches neighboring candidates and
          moves toward patterns with improved support.

        * ``random``:
          Random Search GRAANK. Randomly samples and evaluates gradual pattern
          candidates.

        * ``clustergp``:
          ClusterGP. Uses clustering-based search to identify gradual patterns
          from the transformed dataset.

        Args:
            target_col:
                Index of the target attribute or feature. Temporal transformations
                and time-delay estimation are performed relative to this attribute.

            search_algorithm:
                Gradual pattern mining algorithm to apply to the transformed
                dataset. Supported values are ``apriori``, ``ga``, ``aco``,
                ``pso``, ``hc``, ``random``, and ``clustergp``.
                Defaults to ``"apriori"``.

            max_iteration:
                The maximum number of iterations to run the search algorithm.

            compute_descriptors:
                If ``True``, compute descriptors for the mined gradual patterns.
                Defaults to ``False``.

            ignore_time:
                Mine TGPs but skip the calculation and estimation of time delay.

        Returns:
            A list containing the mined Fuzzy Temporal Gradual Patterns.

        Raises:
            ValueError:
                If ``target_col`` is invalid or ``search_algorithm`` is not one
                of the supported algorithms.

            TypeError:
                If ``target_col`` or another argument has an invalid type.

        Notes:
            Metaheuristic search algorithms such as ``ga``, ``aco``, ``pso``,
            ``hc``, and ``random`` generally provide approximate solutions and
            may not enumerate all frequent gradual patterns. APRIORI provides
            exhaustive level-wise candidate generation subject to the configured
            search constraints.

            Multiprocessing can significantly reduce computation time for large
            datasets, particularly during the temporal transformation and
            evaluation stages.
        """
        start = time.time()
        self.target_col = target_col
        self._search_algorithm = search_algorithm
        self._algorithm_max_iter = max_iteration
        self._compute_descriptors = compute_descriptors
        self.clear_gradual_patterns()

        # 1. Mine FTGPs (using parallel multiprocessing)
        # with mp.Pool(num_cores) as pool:
        #    steps = range(1, self.max_step)
        #    pattern_data = pool.map(self._safe_transform_and_mine, steps)
        pattern_data = []
        feature_cols = self.attr_cols
        for step in range(1, self.max_step):
            transformation_steps: dict = {
                int(feature_cols[i]): step for i in range(len(feature_cols))
            }
            pattern_data.append(
                self._safe_transform_and_mine(
                    transformation_steps, step, skip_time=ignore_time
                )
            )

        # 2. Organize FTGPs into a single list
        for item in pattern_data:
            if item is None:
                continue

            # Standardize 'item' into a list so we only need one loop
            lst_tgp = item if isinstance(item, list) else [item]

            for pat in lst_tgp:
                if isinstance(pat, TGP):
                    self.add_gradual_pattern(pat)

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "TGrad",
            # "Memory Usage (MiB)": f{mem_use)}",
            "GP Search Algorithm": f"{self._search_algorithm}",
            "Maximum Iteration for Search Algorithm": f"{self._algorithm_max_iter}",
            "Minimum Representation": f"{self.min_rep:.2f}",
            "Target Column": f"{target_col}",
            "Run-time": f"{duration:.6f} seconds",
        }
        out_dict.update({"Patterns": self.display_patterns})
        return out_dict

    def transform_data(
        self, transformation_steps: dict, max_step: int
    ) -> tuple[np.ndarray | None, dict]:
        """
        A method that transforms each attribute's data according to a specific transformation step and
        computes the corresponding time-delay values for that attribute.

        :param transformation_steps: A dict that indicates column with its corresponding transformation step--{col1: step, ...,}.
        :param max_step: Largest data transformation step.
        :return: Combined transformed dataset with corresponding time-delay values.
        """

        def get_time_data(curr_step):
            """
            Retrieves the time array for a given step.

            Tries to pull data using the first column of a completed step.
            If the step hasn't been completed, it calculates the fallback time.
            """
            try:
                # Get the list of column indices processed in this step
                completed_cols = completed_steps[curr_step]

                # Fetch the time array associated with the first column of this step
                time_arr = time_data[completed_cols[0]]
            except KeyError:
                # Fallback: Calculate the time array if the step doesn't exist yet
                _, time_arr = self.get_time_diffs(curr_step)
            return time_arr

        if self._time_ok:
            transformed_data: np.ndarray | None = None
            time_data: dict[
                int, np.ndarray
            ] = {}  # {col1: [time-lags], col2: [time-lags]}
            completed_steps: dict[int, list] = {}  # {completed-step: [columns]}
            n = self.row_count
            k = n - max_step  # Number of rows created by the largest step-delay
            for col_index in range(self.col_count):
                if (col_index == self.target_col) or (col_index in self.time_cols):
                    # date-time column OR target column
                    temp_col = self.full_attr_data[col_index][0:k]
                else:
                    # other attributes
                    step = transformation_steps[col_index]
                    temp_col = self.full_attr_data[col_index][step:n]
                    # Get first k items for delayed data
                    temp_col = temp_col[0:k]

                    # Get time-delay values
                    time_data[col_index] = get_time_data(step)[0:k]

                    # Save step so that we do not repeat get_time_diffs for a similar step
                    completed_steps.setdefault(step, []).append(col_index)

                    # for i in range(k):
                    #    if i in time_dict:
                    #        time_dict[i].append(time_diffs[i])
                    #    else:
                    #        time_dict[i] = [time_diffs[i]]
                    # print(f"{time_diffs}\n")
                    # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
                transformed_data = (
                    temp_col
                    if (transformed_data is None)
                    else np.vstack((transformed_data, temp_col))
                )
            return transformed_data, time_data
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise FatalError(msg)

    def _safe_transform_and_mine(
        self, transformation_steps: dict, max_step: int, skip_time: bool = False
    ):
        """Wrapper to catch exceptions during parallel mining."""
        try:
            # 1. Calculate the time difference using a step
            transformed_data, time_data = self.transform_data(
                transformation_steps, max_step
            )
            # 2. Execute GRAANK for each transformation
            if skip_time:
                time_data = None
            t_gps = self._mine_gps_at_step(
                time_delay_data=time_data, transformed_data=transformed_data
            )
            if len(t_gps) > 0:
                return t_gps
            return False
        except FatalError as e:
            print(f"Error at step {max_step}: {e}")
            return None

    def _mine_gps_at_step(
        self, time_delay_data: dict | None, transformed_data: np.ndarray | None = None
    ) -> list[TGP]:
        """
        Uses apriori algorithm to find GP candidates based on the target-attribute. The candidates are validated if
        their computed support is greater than or equal to the minimum support threshold specified by the user.

        :param time_delay_data: Time-delay values
        :param transformed_data: the transformed data.

        :return: Temporal-GPs as a list.
        """

        if transformed_data is None:
            return []

        if time_delay_data is None:
            time_data: dict = {
                "time_data": None,
                "use_gp": False,
                "fuzzy_mfs": [],
                "inference": self.inference_method,
            }
        else:
            t_lag_arr: np.ndarray = np.array(list(time_delay_data.values()))
            fuzzy_mfs = self.build_membership_functions(t_lag_arr)
            time_data: dict = {
                "time_data": time_delay_data,
                "use_gp": True,
                "fuzzy_mfs": fuzzy_mfs,
                "inference": self.inference_method,
            }

        if type(self) is TGrad:
            time_data["use_gp"] = False

        data_df = pd.DataFrame(transformed_data.T, columns=self.titles)
        if data_df.empty:
            return []

        mine_obj = GRAANK(data_df, min_sup=self.thd_supp, eq=self._include_equal_values)
        mine_obj.discover(
            search_type=self._search_algorithm,
            target_col=self.target_col,
            time_data=time_data,
            compute_descriptors=self._compute_descriptors,
            max_iteration=self._algorithm_max_iter,
        )
        return mine_obj.mining_engine.gradual_patterns

    def get_time_diffs(self, step: int) -> tuple[dict, np.ndarray]:  # optimized
        """
        A method that computes the difference between 2 timestamps separated by a specific transformation step.

        :param step: Data transformation step.
        :return: Dict of time delay values
        """
        size = self.row_count
        time_diffs = {}  # {row: time-lag}
        time_diffs_arr = []
        for i in range(size):
            if i < (size - step):
                stamp_1 = 0
                stamp_2 = 0
                for col in self.time_cols:  # sum timestamps from all time-columns
                    time_col_title = self.titles[col]
                    if time_col_title == NO_TIME_LABEL:
                        stamp_1 += int(self.data[i][int(col)])
                        stamp_2 += int(self.data[i + step][int(col)])
                        continue

                    temp_1 = str(self.data[i][int(col)])
                    temp_2 = str(self.data[i + step][int(col)])
                    temp_stamp_1 = TGrad.get_timestamp(temp_1)
                    temp_stamp_2 = TGrad.get_timestamp(temp_2)
                    if (not temp_stamp_1) or (not temp_stamp_2):
                        # Unable to read time
                        msg = f"Error: Time in row {i + 1} or row {i + step + 1} is not valid!"
                        raise ValueError(msg)
                    else:
                        stamp_1 += temp_stamp_1
                        stamp_2 += temp_stamp_2
                time_diff = stamp_2 - stamp_1
                # if time_diff < 0:
                # Error time CANNOT go backwards,
                # print(f"Problem {i} and {i + step} - {self.time_cols}")
                #    return False, [i + 1, i + step + 1]
                time_diff_abs = float(abs(time_diff))
                time_diffs[int(i)] = time_diff_abs
                time_diffs_arr.append(time_diff_abs)
        return time_diffs, np.array(time_diffs_arr)

    def build_membership_functions(
        self,
        time_data: np.ndarray | None,
    ) -> list[dict]:
        """Build membership-function parameters from time-delay data.

        The number of membership functions is estimated from the dominant
        singular-value energy of a trajectory (Hankel) matrix. The time-delay
        values are then clustered using either 1-D K-Means or Fuzzy C-Means
        (FCM), and the resulting cluster centers and spreads are used to
        construct triangular, trapezoidal, or Gaussian membership functions.

        This method intentionally uses NumPy rather than a separate GPU
        implementation. The computations operate on one-dimensional time-delay
        data, so keeping them on the CPU avoids unnecessary CPU-GPU transfers
        and additional memory overhead.

        Args:
            time_data: One-dimensional array containing time-delay values.
                ``None`` returns an empty list.

        Returns:
            A list of membership-function specifications. Each specification
            contains ``shape`` and ``params`` keys.

        Raises:
            ValueError: If no finite values are available or an unsupported
                membership-function shape is specified.
        """
        if time_data is None:
            return []

        values = np.asarray(time_data, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]

        if values.size == 0:
            return []

        def estimate_n_clusters(threshold: float = 0.90) -> int:
            """Estimate the number of latent temporal components."""
            total_count = values.size

            if total_count < 3:
                return 2

            threshold = float(np.clip(threshold, 0.0, 1.0))
            window_len = total_count // 2
            hankel_mat = np.lib.stride_tricks.sliding_window_view(values, window_len).T

            singular_values = np.linalg.svd(hankel_mat, compute_uv=False)
            energy = singular_values**2
            total_energy = energy.sum()

            if total_energy <= np.finfo(float).eps:
                return 2

            cumulative_energy = np.cumsum(energy) / total_energy
            estimated = (
                np.searchsorted(
                    cumulative_energy,
                    threshold,
                    side="left",
                )
                + 1
            )

            return max(2, int(estimated))

        def compute_kmeans(
            data: np.ndarray,
            n_clusters: int,
            max_iter: int = 100,
            tolerance: float = 1e-6,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Perform vectorized one-dimensional K-Means clustering."""
            minimum, maximum = data.min(), data.max()
            centers = np.linspace(
                minimum,
                maximum,
                n_clusters,
                dtype=np.float64,
            )
            for _ in range(max_iter):
                distances = np.abs(data[:, None] - centers[None, :])
                labels = np.argmin(
                    distances,
                    axis=1,
                )
                counts = np.bincount(
                    labels,
                    minlength=n_clusters,
                ).astype(np.float64)
                sums = np.bincount(
                    labels,
                    weights=data,
                    minlength=n_clusters,
                )

                new_centers = centers.copy()
                non_empty = counts > 0
                new_centers[non_empty] = sums[non_empty] / counts[non_empty]
                if np.allclose(
                    centers,
                    new_centers,
                    rtol=tolerance,
                    atol=tolerance,
                ):
                    centers = new_centers
                    break
                centers = new_centers

            order = np.argsort(centers)
            centers = centers[order]

            # Reassign after sorting to obtain correctly matched spreads.
            labels = np.argmin(
                np.abs(data[:, None] - centers[None, :]),
                axis=1,
            )
            counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
            sums = np.bincount(labels, weights=data, minlength=n_clusters)
            squared_sums = np.bincount(labels, weights=data**2, minlength=n_clusters)
            spreads = np.zeros(n_clusters, dtype=np.float64)

            non_empty = counts > 0
            means = np.zeros(
                n_clusters,
                dtype=np.float64,
            )
            means[non_empty] = sums[non_empty] / counts[non_empty]

            variances = np.zeros(n_clusters, dtype=np.float64)
            variances[non_empty] = (
                squared_sums[non_empty] / counts[non_empty] - means[non_empty] ** 2
            )
            spreads[non_empty] = np.sqrt(np.maximum(variances[non_empty], 0.0))

            fallback_spread = max(
                float(np.std(data)),
                0.1,
            )
            spreads[~non_empty] = fallback_spread
            return centers, spreads

        def compute_fcm(
            data: np.ndarray,
            n_clusters: int,
            fuzziness: float = 2.0,
            max_iter: int = 100,
            tolerance: float = 1e-6,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Perform vectorized one-dimensional Fuzzy C-Means clustering."""
            minimum, maximum = data.min(), data.max()
            centers = np.linspace(
                minimum,
                maximum,
                n_clusters,
                dtype=np.float64,
            )
            exponent = 2.0 / (fuzziness - 1.0)
            eps = np.finfo(np.float64).eps
            membership_m = np.array([])
            for _ in range(max_iter):
                distances = np.abs(data[:, None] - centers[None, :])
                zero_distance = distances <= eps
                safe_distances = np.maximum(distances, eps)
                weights = safe_distances ** (-exponent)

                zero_rows = zero_distance.any(axis=1)
                if np.any(zero_rows):
                    weights[zero_rows] = zero_distance[zero_rows]

                membership = weights / weights.sum(axis=1, keepdims=True)
                membership_m = membership**fuzziness
                denominator = membership_m.sum(axis=0)

                new_centers = (membership_m * data[:, None]).sum(axis=0) / np.maximum(
                    denominator, eps
                )
                if np.allclose(centers, new_centers, rtol=tolerance, atol=tolerance):
                    centers = new_centers
                    break
                centers = new_centers

            order = np.argsort(centers)
            centers = centers[order]
            membership_m = membership_m[:, order]
            denominator = membership_m.sum(axis=0)

            spreads = np.sqrt(
                np.maximum(
                    (membership_m * (data[:, None] - centers[None, :]) ** 2).sum(axis=0)
                    / np.maximum(denominator, eps),
                    0.0,
                )
            )

            fallback_spread = max(float(np.std(data)), 0.1)
            spreads = np.where(
                np.isfinite(spreads) & (spreads > 0.0),
                spreads,
                fallback_spread,
            )

            return centers, spreads

        # --------------------------------------------------------------
        # Estimate number of membership functions.
        # --------------------------------------------------------------
        num_clusters = estimate_n_clusters()

        # Avoid requesting more clusters than distinct values.
        num_clusters = min(
            num_clusters,
            max(2, np.unique(values).size),
        )

        # --------------------------------------------------------------
        # Cluster time-delay data.
        # --------------------------------------------------------------
        min_val, max_val = values.min(), values.max()
        if np.isclose(min_val, max_val):
            peaks = np.full(
                num_clusters,
                min_val,
                dtype=np.float64,
            )
            bounds = np.full(
                num_clusters,
                0.1,
                dtype=np.float64,
            )
        elif self.clustering_algorithm.lower() == "kmeans":
            peaks, bounds = compute_kmeans(
                values,
                num_clusters,
            )
        else:
            peaks, bounds = compute_fcm(
                values,
                num_clusters,
            )

        # --------------------------------------------------------------
        # Build membership functions.
        # --------------------------------------------------------------
        shape = self.mf_shape.lower()

        if shape not in {
            "triangular",
            "trapezoidal",
            "gaussian",
        }:
            raise ValueError(
                f"Unsupported membership-function shape: {self.mf_shape!r}. Expected 'triangular', "
                "'trapezoidal', or 'gaussian'."
            )

        mf_params: list[dict] = []

        for i, (center, spread) in enumerate(zip(peaks, bounds)):
            center = float(center)
            spread = max(float(spread), 0.1)

            if shape == "triangular":
                left = float(peaks[i - 1]) if i > 0 else center - 3.0 * spread
                right = (
                    float(peaks[i + 1])
                    if i < num_clusters - 1
                    else center + 3.0 * spread
                )

                mf_params.append(
                    {
                        "shape": "triangular",
                        "params": [left, center, right],
                    }
                )

            elif shape == "trapezoidal":
                left = float(peaks[i - 1]) if i > 0 else center - 4.0 * spread
                right = (
                    float(peaks[i + 1])
                    if i < num_clusters - 1
                    else center + 4.0 * spread
                )

                mf_params.append(
                    {
                        "shape": "trapezoidal",
                        "params": [
                            left,
                            center - 0.5 * spread,
                            center + 0.5 * spread,
                            right,
                        ],
                    }
                )

            else:
                mf_params.append(
                    {
                        "shape": "gaussian",
                        "params": [center, spread],
                    }
                )

        return mf_params

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
