# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@version: 0.3.8
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 27 October 2022

SO4GP
------

    **SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm
    intelligence, HDF5 chunks, SVD and many others in order to improve the efficiency of extracting gradual patterns
    (GPs). A GP is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is a pair
    (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
    GIs; for example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

    The classical approach for mining GPs is computationally expensive. This package provides Python algorithm
    implementations of several optimization techniques that are applied to the classical approach in order to improve
    its computational efficiency. The algorithm implementations include:
        * (Classical) GRAANK algorithm for extracting GPs
        * Ant Colony Optimization algorithm for extracting GPs
        * Genetic Algorithm for extracting GPs
        * Particle Swarm Optimization algorithm for extracting GPs
        * Random Search algorithm for extracting GPs
        * Local Search algorithm for extracting GPs

    Apart from swarm-based optimization techniques, this package also provides a Python algorithm implementation of a
    clustering approach for mining GPs.

"""

# from collections import defaultdict
# from dateutil.parser import parse
import time
import gc
import math
import json
import random
import numpy as np
from ypstruct import structure
from sklearn.cluster import KMeans

from __configs__ import *
from . import DataGP, GI, ExtGP, NumericSS


# --------- ALGORITHMS ---------------------


class AntGRAANK(DataGP):
    """Description of class AntGRAANK

    Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization approach
    (proposed in a published paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is
    measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

        In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid
        (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or
        decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of
        columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by
        randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that
        its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the
        pheromone levels and better candidates are generated.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        evaporation_factor: value between 0-1 which determines how fast pheromone levels evaporate

        distance_matrix: an array that stores the cost between travelling between nodes

        attribute_keys: an array with attribute keys

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "ACO-GRAANK", "Best Patterns": [[["Expenses-", "Age+"], 1.0]], "Invalid Count": 1, "Iterations": 3}

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, e_factor=EVAPORATION_FACTOR):
        """Description

    Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization approach
    (proposed in a published paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is
    measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

        In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid
        (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or
        decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of
        columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by
        randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that
        its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the
        pheromone levels and better candidates are generated.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        evaporation_factor: value between 0-1 which determines how fast pheromone levels evaporate

        distance_matrix: an array that stores the cost between travelling between nodes

        attribute_keys: an array with attribute keys

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "ACO-GRAANK", "Best Patterns": [[["Expenses-", "Age+"], 1.0]], "Invalid Count": 1, "Iterations":3}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param e_factor: evaporation factor, default is 0.5

        """
        super(AntGRAANK, self).__init__(*args)
        self.evaporation_factor = e_factor
        self.max_iteration = max_iter
        self.distance_matrix = None
        self.attribute_keys = None

    def _fit(self):
        """Description

        Generates the distance matrix (d)
        :return: distance matrix (d) and attribute keys
        """
        v_bins = self.valid_bins
        # 1. Fetch valid bins group
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in v_bins[:, 0]]

        # 2. Initialize an empty d-matrix
        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
        for i in range(n):
            for j in range(n):
                if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                    # Ignore similar attributes (+ or/and -)
                    continue
                else:
                    bin_1 = v_bins[i][1]
                    bin_2 = v_bins[j][1]
                    # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                    d[i][j] += np.sum(np.multiply(bin_1, bin_2))
        # print(d)
        self.distance_matrix = d
        self.attribute_keys = attr_keys
        gc.collect()

    def _gen_aco_candidates(self, p_matrix):
        """Description

        Generates GP candidates based on the pheromone levels.

        :param p_matrix: pheromone matrix (ndarray)
        :return: pheromone matrix (ndarray)
        """
        v_matrix = self.distance_matrix
        pattern = ExtGP()
        ":type pattern: ExtGP"

        # 1. Generate gradual items with the highest pheromone and visibility
        m = p_matrix.shape[0]
        for i in range(m):
            combine_feature = np.multiply(v_matrix[i], p_matrix[i])
            total = np.sum(combine_feature)
            with np.errstate(divide='ignore', invalid='ignore'):
                probability = combine_feature / total
            cum_prob = np.cumsum(probability)
            r = np.random.random_sample()
            try:
                j = np.nonzero(cum_prob > r)[0][0]
                gi = GI.parse_gi(self.attribute_keys[j])
                """:type gi: GI"""
                if not pattern.contains_attr(gi):
                    pattern.add_gradual_item(gi)
            except IndexError:
                continue

        # 2. Evaporate pheromones by factor e
        p_matrix = (1 - self.evaporation_factor) * p_matrix
        return pattern, p_matrix

    def _update_pheromones(self, pattern, p_matrix):
        """Description

        Updates the pheromone level of the pheromone matrix

        :param pattern: pattern used to update values
        :param p_matrix: an existing pheromone matrix
        :return: updated pheromone matrix
        """
        idx = [self.attribute_keys.index(x.as_string()) for x in pattern.gradual_items]
        for n in range(len(idx)):
            for m in range(n + 1, len(idx)):
                i = idx[n]
                j = idx[m]
                p_matrix[i][j] += 1
                p_matrix[j][i] += 1
        return p_matrix

    def discover(self):
        """Description

        Applies ant-colony optimization algorithm and uses pheromone levels to find GP candidates. The candidates are
        validated if their computed support is greater than or equal to the minimum support threshold specified by the
        user.

        :return: JSON object
        """
        # 0. Initialize and prepare data set
        # d_set = DataGP(f_path, min_supp)
        # """:type d_set: DataGP"""
        self.fit_bitmap()
        self._fit()  # distance matrix (d) & attributes corresponding to d
        d = self.distance_matrix

        a = self.attr_size
        self.gradual_patterns = list()  # subsets
        loser_gps = list()  # supersets
        str_winner_gps = list()  # subsets
        repeated = 0
        it_count = 0
        counter = 0

        if self.no_bins:
            return []

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((self.thd_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0

        # 3. Initialize pheromones (p_matrix)
        pheromones = np.ones(d.shape, dtype=float)

        invalid_count = 0
        # 4. Iterations for ACO
        # while repeated < 1:
        while counter < self.max_iteration:
            rand_gp, pheromones = self._gen_aco_candidates(pheromones)
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = rand_gp.is_duplicate(self.gradual_patterns, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = rand_gp.check_am(loser_gps, subset=False)
                    is_sub = rand_gp.check_am(self.gradual_patterns, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = rand_gp.validate_graank(self)
                    """:type gen_gp: ExtGP"""
                    is_present = gen_gp.is_duplicate(self.gradual_patterns, loser_gps)
                    is_sub = gen_gp.check_am(self.gradual_patterns, subset=True)
                    if is_present or is_sub:
                        repeated += 1
                    else:
                        if gen_gp.support >= self.thd_supp:
                            pheromones = self._update_pheromones(gen_gp, pheromones)
                            self.gradual_patterns.append(gen_gp)
                            str_winner_gps.append(gen_gp.print(self.titles))
                        else:
                            loser_gps.append(gen_gp)
                            invalid_count += 1
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
            else:
                invalid_count += 1
            it_count += 1
            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "ACO-GRAANK", "Best Patterns": str_winner_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        return out


class ClusterGP(DataGP):
    """Description of class CluDataGP (Clustering DataGP)

    CluDataGP stands for Clustering DataGP. It is a class that inherits the DataGP class in order to create data-gp
    objects for the clustering approach. This class inherits the DataGP class which is used to create data-gp objects.
    The classical data-gp object is meant to store all the parameters required by GP algorithms to extract gradual
    patterns (GP). It takes a numeric file (in CSV format) as input and converts it into an object whose attributes are
    used by algorithms to extract GPs.

    class DataGP provides the following attributes:
        thd_supp: minimum support threshold

        equal: eq value

        titles: column names of data source

        data: all the objects organized into their respective column

        row_count: number of objects

        col_count: number of all columns

        time_cols: column indices of the columns with data-time objects

        attr_cols: column indices of the columns with numeric values

        valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols, a bitmap
        is valid if its computed support is equal or greater than the minimum support threshold

        no_bins: True if all none of the attr_cols yields a valid bitmap

        gradual_patterns: list of GP objects

    This class adds the parameters required for clustering gradual items to the data-gp object. The class provides the
    following additional attributes:
        e_prob: erasure probability (a value between 0 - 1)

        mat_iter: maximum iteration value for score vector estimation

    CluDataGP adds the following functions:
        construct_matrices: generates the net-win matrix

        infer_gps: infers GPs from clusters of Gradual Items

        estimate_score_vector: estimates the score vector based on the cumulative wins

        estimate_support:  estimates the frequency support of a GP based on its score vector

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "Clu-GRAANK", "Patterns": [[["Age-", "Expenses+"], 0.8]], "Invalid Count": 0}

    """

    def __init__(self, *args, e_prob=ERASURE_PROBABILITY, max_iter=SCORE_VECTOR_ITERATIONS, no_prob=False):
        """Description of class CluDataGP (Clustering DataGP)

        A class for creating data-gp objects for the clustering approach. This class inherits the DataGP class which is
        used to create data-gp objects. This class adds the parameters required for clustering gradual items to the
        data-gp object.

        The class provides the following additional attributes:

            e_prob: erasure probability (a value between 0 - 1). Erasure probability determines the proportion of ij
            columns to be used by the algorithm, the rest are ignored.

            mat_iter: maximum iteration value for score vector estimation

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5, no_prob=True)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP

        :param args: [required] data-source, [optional] minimum-support
        :param e_prob: [optional] erasure probability, the default is 0.5
        :param max_iter: [optional] maximum iteration for score vector estimation, the default is 10
        """
        super(ClusterGP, self).__init__(*args)
        self.erasure_probability = e_prob
        """:type erasure_probability: float"""
        self.max_iteration = max_iter
        """:type max_iteration: int"""
        if not no_prob:
            self.gradual_items, self.cum_wins, self.net_win_mat, self.ij = self._construct_matrices(e_prob)
            """:type gradual_items: np.ndarray"""
            """:type cum_wins: np.ndarray"""
            """:type net_win_mat: np.ndarray"""
            """:type ij: np.ndarray"""
            self.win_mat = np.array([])
            """:type win_mat: np.ndarray"""
        else:
            self.gradual_items, self.win_mat, self.cum_wins, self.net_win_mat, self.ij = self._construct_all_matrices()
            """:type gradual_items: np.ndarray"""
            """:type win_mat: np.ndarray"""
            """:type cum_wins: np.ndarray"""
            """:type net_win_mat: np.ndarray"""
            # """:type nodes_mat: np.ndarray"""
            """:type ij: np.ndarray"""

    def _construct_matrices(self, e):
        """Description

        Generates all the gradual items and, constructs: (1) net-win matrix, (2) cumulative wins, (3) pairwise objects.

        :param e: [required] erasure probability
        :type e: float

        :return: list of gradual items, net-win matrix, cumulative win matrix, selected pairwise (ij) objects
        """

        n = self.row_count
        prob = 1 - e  # Sample probability

        if prob == 1:
            # 1a. Generate all possible pairs
            pair_ij = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

            # 1b. Remove duplicates or reversed pairs
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] < pair_ij[:, 1])[:, 0]]
        else:
            # 1a. Generate random pairs using erasure-probability
            total_pair_count = int(n * (n - 1) * 0.5)
            rand_1d = np.random.choice(n, int(prob * total_pair_count) * 2, replace=True)
            pair_ij = np.reshape(rand_1d, (-1, 2))

            # 1b. Remove duplicates
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] != pair_ij[:, 1])[:, 0]]

        # 2. Variable declarations
        attr_data = self.data.T  # Feature data objects
        lst_gis = []  # List of GIs
        s_mat = []  # S-Matrix (made up of S-Vectors)
        cum_wins = []  # Cumulative wins

        # 3. Construct S matrix from data set
        for col in np.nditer(self.attr_cols):
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))
            # print(col)
            # print(temp_cum_wins)

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # i wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # j loses/wins (1/-1)
            # print(s_vec)
            # print("\n")
            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                lst_gis.append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                lst_gis.append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        return np.array(lst_gis), np.array(cum_wins), np.array(s_mat), pair_ij

    def _construct_all_matrices(self):
        """Description

        Generates all the gradual items and, constructs: (1) win matrix (2) net-win matrix, (3) cumulative wins,
        (4) pairwise objects.

        :return: list of gradual items, win matrix, net-win matrix, cumulative win matrix, selected (ij) objects
        """

        n = self.row_count

        # 1a. Generate all possible pairs
        pair_ij = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

        # 1b. Remove duplicates or reversed pairs
        pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] < pair_ij[:, 1])[:, 0]]

        # 2. Variable declarations
        attr_data = self.data.T  # Feature data objects
        lst_gis = []  # List of GIs
        s_mat = []  # S-Matrix (made up of S-Vectors)
        w_mat = []  # win matrix
        cum_wins = []  # Cumulative wins
        # nodes_mat = []  # FP nodes matrix

        # 3. Construct S matrix from data set
        for col in np.nditer(self.attr_cols):
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            # nodes_vec = [[set(), set()]] * n
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # i wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # j loses/wins (1/-1)

                """
                if w == 1:
                    for node_i in i:
                        nodes_j = j[np.where(j > node_i)]
                        tmp = nodes_vec[node_i][0].union(set(nodes_j))
                        nodes_vec[node_i] = [tmp, nodes_vec[node_i][1]]

                    for node_j in j:
                        nodes_i = i[np.where(i < node_j)]
                        tmp = nodes_vec[node_j][1].union(set(nodes_i))
                        nodes_vec[node_j] = [nodes_vec[node_j][0], tmp]
                elif w == -1:
                    for node_i in i:
                        nodes_j = j[np.where(j > node_i)]
                        tmp = nodes_vec[node_i][1].union(set(nodes_j))
                        nodes_vec[node_i] = [nodes_vec[node_i][0], tmp]

                    for node_j in j:
                        nodes_i = i[np.where(i < node_j)]
                        tmp = nodes_vec[node_j][0].union(set(nodes_i))
                        nodes_vec[node_j] = [tmp, nodes_vec[node_j][1]]

            # print('positions: ' + str(positions) + '; i: ' + str(i) + '; j: ' + str(j) + '; counts: ' + str(counts_i))
            #    print(nodes_vec)
            # print("\n")"""

            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                w_mat.append(np.copy(s_vec))
                # nodes_mat.append(nodes_vec)

                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                lst_gis.append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                lst_gis.append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        # print(np.array(nodes_mat))
        return np.array(lst_gis), np.array(w_mat), np.array(cum_wins), np.array(s_mat), pair_ij

    def _infer_gps(self, clusters):
        """Description

        A function that infers GPs from clusters of gradual items.

        :param clusters: [required] groups of gradual items clustered through K-MEANS algorithm
        :type clusters: np.ndarray

        :return: list of (str) patterns, list of GP objects
        """

        patterns = []
        str_patterns = []

        all_gis = self.gradual_items
        cum_wins = self.cum_wins

        lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
        for grp_idx in lst_indices:
            if grp_idx.size > 1:
                # 1. Retrieve all cluster-pairs and the corresponding GIs
                cluster_gis = all_gis[grp_idx]
                cluster_cum_wins = cum_wins[grp_idx]  # All the rows of selected groups

                # 2. Compute score vector from R matrix
                score_vectors = []  # Approach 2
                for c_win in cluster_cum_wins:
                    temp = self._estimate_score_vector(c_win)
                    score_vectors.append(temp)

                # 3. Estimate support
                est_sup = self._estimate_support(score_vectors)

                # 4. Infer GPs from the clusters
                if est_sup >= self.thd_supp:
                    gp = ExtGP()
                    for gi in cluster_gis:
                        gp.add_gradual_item(gi)
                    gp.set_support(est_sup)
                    patterns.append(gp)
                    str_patterns.append(gp.print(self.titles))
        return str_patterns, patterns

    def _estimate_score_vector(self, c_wins):
        """Description

        A function that estimates the score vector based on the cumulative wins.

        :param c_wins: [required] cumulative wins
        :type c_wins: np.ndarray

        :return: score vector (ndarray)
        """

        # Estimate score vector from pairs
        n = self.row_count
        score_vector = np.ones(shape=(n,))
        arr_ij = self.ij

        # Construct a win-matrix
        temp_vec = np.zeros(shape=(n,))
        pair_count = arr_ij.shape[0]

        # Compute score vector
        for _ in range(self.max_iteration):
            if np.count_nonzero(score_vector == 0) > 1:
                break
            else:
                for pr in range(pair_count):
                    pr_val = c_wins[pr]
                    i = arr_ij[pr][0]
                    j = arr_ij[pr][1]
                    if pr_val == 1:
                        log = math.log(
                            math.exp(score_vector[i]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                            10)
                        temp_vec[i] += pr_val * log
                    elif pr_val == -1:
                        log = math.log(
                            math.exp(score_vector[j]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                            10)
                        temp_vec[j] += -pr_val * log
                score_vector = abs(temp_vec / np.sum(temp_vec))
        return score_vector

    def _estimate_support(self, score_vectors):
        """Description

        A function that estimates the frequency support of a GP based on its score vector.

        :param score_vectors: score vector (ndarray)
        :return: estimated support (float)
        """

        # Estimate support - use different score-vectors to construct pairs
        n = self.row_count
        bin_mat = np.ones((n, n), dtype=np.bool)
        for vec in score_vectors:
            temp_bin = vec < vec[:, np.newaxis]
            bin_mat = np.multiply(bin_mat, temp_bin)

        est_sup = float(np.sum(bin_mat)) / float(n * (n - 1.0) / 2.0)
        """:type est_sup: float"""
        return est_sup

    def discover(self, testing=False):
        """Description

        Applies spectral clustering to determine which gradual items belong to the same group based on the similarity
        of net-win vectors. Gradual items in the same cluster should have almost similar score vector. The candidates
        are validated if their computed support is greater than or equal to the minimum support threshold specified by
        the user.

        :param testing: [optional] returns different format if algorithm is used in a test environment
        :type testing: bool

        :return: JSON object
        """

        # 1. Generate net-win matrices
        s_matrix = self.net_win_mat  # Net-win matrix (S)
        if s_matrix.size < 1:
            raise Exception("Erasure probability is too high, consider reducing it.")
        # print(s_matrix)

        start = time.time()  # TO BE REMOVED
        # 2a. Spectral Clustering: perform SVD to determine the independent rows
        u, s, vt = np.linalg.svd(s_matrix)

        # 2b. Spectral Clustering: compute rank of net-wins matrix
        r = np.linalg.matrix_rank(s_matrix)  # approximated r

        # 2c. Spectral Clustering: rank approximation
        s_matrix_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

        # 2d. Clustering using K-Means (using sklearn library)
        kmeans = KMeans(n_clusters=r, random_state=0)
        y_predicted = kmeans.fit_predict(s_matrix_approx)

        end = time.time()  # TO BE REMOVED

        # 3. Infer GPs
        str_gps, estimated_gps = self._infer_gps(y_predicted)

        # 4. Output - DO NOT ADD TO PyPi Package
        out = structure()
        out.estimated_gps = estimated_gps
        out.max_iteration = self.max_iteration
        out.titles = self.titles
        out.col_count = self.col_count
        out.row_count = self.row_count
        out.e_prob = self.erasure_probability
        out.cluster_time = (end - start)  # TO BE REMOVED
        if testing:
            return out

        # Output
        out = json.dumps({"Algorithm": "Clu-GRAANK", "Patterns": str_gps, "Invalid Count": 0})
        """:type out: object"""
        self.gradual_patterns = estimated_gps
        return out


class GeneticGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm approach (proposed
    in a published  paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is measured by
    its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects.
    A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
    values of column age 'increasing' and column 'salary' decreasing.

         In this approach, we assume that every GP candidate may be represented as a binary gene (or individual) that
         has a unique position and cost. The cost is derived from the computed support of that candidate, the higher the
         support value the lower the cost. The aim of the algorithm is search through a population of individuals (or
         candidates) and find those with the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        n_pop: integer value that determines the initial population size of individuals

        pc: a value that determines the proportion of children

        gamma: a value in the range 0-1 that determines the cross-over rate

        mu: a value in the range 0-1 that determines the mutation rate

        sigma: a value in the range 0-1 that determines the mutation rate

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.GeneticGRAANK(dummy_df, 0.5, max_iter=3, n_pop=10)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "GA-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 12,
    "Iterations": 2}

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, n_pop=N_POPULATION, pc=PC, gamma=GAMMA, mu=MU, sigma=SIGMA):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm approach (proposed
        in a published  paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is measured by
        its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects.
        A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
        values of column age 'increasing' and column 'salary' decreasing.

             In this approach, we assume that every GP candidate may be represented as a binary gene (or individual)
             that has a unique position and cost. The cost is derived from the computed support of that candidate, the
             higher the support value the lower the cost. The aim of the algorithm is search through a population of
             individuals (or candidates) and find those with the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            n_pop: integer value that determines the initial population size of individuals

            pc: a value that determines the proportion of children

            gamma: a value in the range 0-1 that determines the cross-over rate

            mu: a value in the range 0-1 that determines the mutation rate

            sigma: a value in the range 0-1 that determines the mutation rate

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GeneticGRAANK(dummy_df, 0.5, max_iter=3, n_pop=10)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "GA-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 12,
        "Iterations": 2}


        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :type max_iter: int

        :param n_pop: initial individual population, default is 5
        :type n_pop: int

        :param pc: children proportion, default is 0.5
        :type pc: float

        :param gamma: cross-over gamma ratio, default is 1
        :type gamma: float

        :param mu: mutation mu ratio, default is 0.9
        :type mu: float

        :param sigma: mutation sigma ratio, default is 0.9
        :type sigma: float
        """
        super(GeneticGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        self.n_pop = n_pop
        self.pc = pc
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma

    def _crossover(self, p1, p2):
        """Description

        Crosses over the genes of 2 parents (an individual with a specific position and cost) in order to generate 2
        different offsprings.

        :param p1: parent 1 individual
        :param p2: parent 2 individual
        :return: 2 offsprings (children)
        """
        c1 = p1.deepcopy()
        c2 = p2.deepcopy()
        alpha = np.random.uniform(0, self.gamma, 1)
        c1.position = alpha * p1.position + (1 - alpha) * p2.position
        c2.position = alpha * p2.position + (1 - alpha) * p1.position
        return c1, c2

    def _mutate(self, x):
        """Description

        Mutates an individual's position in order to create a new and different individual.

        :param x: existing individual
        :return: new individual
        """
        y = x.deepcopy()
        str_x = str(int(y.position))
        flag = np.random.rand(*(len(str_x),)) <= self.mu
        ind = np.argwhere(flag)
        str_y = "0"
        for i in ind:
            val = float(str_x[i[0]])
            val += self.sigma * np.random.uniform(0, 1, 1)
            if i[0] == 0:
                str_y = "".join(("", "{}".format(int(val)), str_x[1:]))
            else:
                str_y = "".join((str_x[:i[0] - 1], "{}".format(int(val)), str_x[i[0]:]))
            str_x = str_y
        y.position = int(str_y)
        return y

    def discover(self):
        """Description

        Uses genetic algorithm to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Problem Information
        # cost_function

        # Parameters
        # pc: Proportion of children (if its 1, then nc == npop
        it_count = 0
        eval_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)

        nc = int(np.round(self.pc * self.n_pop / 2) * 2)  # No. of children np.round is used to get even number

        # Empty Individual Template
        empty_individual = structure()
        empty_individual.position = None
        empty_individual.cost = None

        # Initialize Population
        pop = empty_individual.repeat(self.n_pop)
        for i in range(self.n_pop):
            pop[i].position = random.randrange(var_min, var_max)
            pop[i].cost = 1  # cost_function(pop[i].position, attr_keys, d_set)
            # if pop[i].cost < best_sol.cost:
            #    best_sol = pop[i].deepcopy()

        # Best Solution Ever Found
        best_sol = empty_individual.deepcopy()
        best_sol.position = pop[0].position
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = list()
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:

            c_pop = []  # Children population
            for _ in range(nc // 2):
                # Select Parents
                q = np.random.permutation(self.n_pop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]

                # a. Perform Crossover
                c1, c2 = self._crossover(p1, p2)

                # Apply Bound
                NumericSS.apply_bound(c1, var_min, var_max)
                NumericSS.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = NumericSS.cost_function(c1.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = NumericSS.cost_function(c2.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # b. Perform Mutation
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                # Apply Bound
                NumericSS.apply_bound(c1, var_min, var_max)
                NumericSS.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = NumericSS.cost_function(c1.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = NumericSS.cost_function(c2.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # c. Add Offsprings to c_pop
                c_pop.append(c1)
                c_pop.append(c2)

            # Merge, Sort and Select
            pop += c_pop
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:self.n_pop]

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_sol.cost = 1

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "GA-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out


class GRAANK(DataGP):
    """Description

        Extracts gradual patterns (GPs) from a numeric data source using the GRAANK approach (proposed in a published
        research paper by Anne Laurent).

             A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example
             given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
             with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
             column 'salary' decreasing.

        This class extends class DataGP which is responsible for generating the GP bitmaps.

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "GRAANK", "Patterns": [[[["Expenses-", "Age+"], 1.0], [["Age-", "Salary-"],
         0.6], [["Age-", "Expenses+"], 1.0], [["Expenses-", "Salary+"], 0.6], [["Salary-", "Expenses+"], 0.6],
         [["Expenses-", "Age+", "Salary+"], 0.6], [["Age-", "Salary-", "Expenses+"], 0.6]], "Invalid Count": 22}

        """

    def _gen_apriori_candidates(self, gi_bins):
        """Description

        Generates Apriori GP candidates
        :param gi_bins: GI together with bitmaps
        :return:
        """
        sup = self.thd_supp
        n = self.row_count

        invalid_count = 0
        res = []
        all_candidates = []
        if len(gi_bins) < 2:
            return []
        try:
            set_gi = [{x[0]} for x in gi_bins]
        except TypeError:
            set_gi = [set(x[0]) for x in gi_bins]

        for i in range(len(gi_bins) - 1):
            for j in range(i + 1, len(gi_bins)):
                try:
                    gi_i = {gi_bins[i][0]}
                    gi_j = {gi_bins[j][0]}
                    gi_o = {gi_bins[0][0]}
                except TypeError:
                    gi_i = set(gi_bins[i][0])
                    gi_j = set(gi_bins[j][0])
                    gi_o = set(gi_bins[0][0])
                gp_cand = gi_i | gi_j
                inv_gp_cand = {GI.inv_arr(x) for x in gp_cand}
                if (len(gp_cand) == len(gi_o) + 1) and (not (all_candidates != [] and gp_cand in all_candidates)) \
                        and (not (all_candidates != [] and inv_gp_cand in all_candidates)):
                    test = 1
                    for k in gp_cand:
                        try:
                            k_set = {k}
                        except TypeError:
                            k_set = set(k)
                        gp_cand_2 = gp_cand - k_set
                        inv_gp_cand_2 = {GI.inv_arr(x) for x in gp_cand_2}
                        if gp_cand_2 not in set_gi and inv_gp_cand_2 not in set_gi:
                            test = 0
                            break
                    if test == 1:
                        m = gi_bins[i][1] * gi_bins[j][1]
                        t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                        if t > sup:
                            res.append([gp_cand, m])
                        else:
                            invalid_count += 1
                    all_candidates.append(gp_cand)
                    gc.collect()
        return res, invalid_count

    def discover(self):
        """Description

        Uses apriori algorithm to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        self.fit_bitmap()

        self.gradual_patterns = []
        """:type patterns: GP list"""
        str_winner_gps = []
        n = self.attr_size
        valid_bins = self.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins)
            invalid_count += inv_count
            i = 0
            while i < len(valid_bins) and valid_bins != []:
                gi_tuple = valid_bins[i][0]
                bin_data = valid_bins[i][1]
                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
                if sup < self.thd_supp:
                    del valid_bins[i]
                    invalid_count += 1
                else:
                    z = 0
                    while z < (len(self.gradual_patterns) - 1):
                        if set(self.gradual_patterns[z].get_pattern()).issubset(set(gi_tuple)):
                            del self.gradual_patterns[z]
                        else:
                            z = z + 1

                    gp = ExtGP()
                    """:type gp: ExtGP"""
                    for obj in valid_bins[i][0]:
                        gi = GI(obj[0], obj[1].decode())
                        """:type gi: GI"""
                        gp.add_gradual_item(gi)
                    gp.set_support(sup)
                    self.gradual_patterns.append(gp)
                    str_winner_gps.append(gp.print(self.titles))
                    i += 1
        # Output
        out = json.dumps({"Algorithm": "GRAANK", "Patterns": str_winner_gps, "Invalid Count": invalid_count})
        """:type out: object"""
        return out


class HillClimbingGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
         associated with it. The cost is derived from the computed support of that candidate, the higher the support
         value the lower the cost. The aim of the algorithm is search through group of positions and find those with
         the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        step_size: integer value that steps the algorithm takes per iteration

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.HillClimbingGRAANK(dummy_df, 0.5, max_iter=3, step_size=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "LS-GRAANK", "Best Patterns": [[["Age+", "Expenses-"], 1.0]], "Invalid Count": 2, "Iterations": 2}

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, step_size=STEP_SIZE):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

             In this approach, we assume that every GP candidate may be represented as a position that has cost value
             associated with it. The cost is derived from the computed support of that candidate, the higher the support
             value the lower the cost. The aim of the algorithm is search through group of positions and find those with
             the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            step_size: integer value that steps the algorithm takes per iteration

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.HillClimbingGRAANK(dummy_df, 0.5, max_iter=3, step_size=0.5)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "LS-GRAANK", "Best Patterns": [[["Age+", "Expenses-"], 1.0]], "Invalid Count": 2, "Iterations": 2}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param step_size: step size, default is 0.5
        """
        super(HillClimbingGRAANK, self).__init__(*args)
        self.step_size = step_size
        self.max_iteration = max_iter

    def discover(self):
        """Description

        Uses hill-climbing algorithm to find GP candidates. The candidates are validated if their computed support is
        greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """
        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Parameters
        it_count = 0
        var_min = 0
        counter = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        eval_count = 0

        # Empty Individual Template
        best_sol = structure()
        candidate = structure()

        # generate an initial point
        best_sol.position = None
        # candidate.position = None
        if best_sol.position is None:
            best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
        # evaluate the initial point
        NumericSS.apply_bound(best_sol, var_min, var_max)
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        # run the hill climb
        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # take a step
            candidate.position = None
            if candidate.position is None:
                candidate.position = best_sol.position + (random.randrange(var_min, var_max) * self.step_size)
            NumericSS.apply_bound(candidate, var_min, var_max)
            candidate.cost = NumericSS.cost_function(candidate.position, attr_keys, self)
            if candidate.cost == 1:
                invalid_count += 1

            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "LS-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out


class ParticleGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
         position and fitness. The fitness is derived from the computed support of that candidate, the higher the
         support value the higher the fitness. The aim of the algorithm is search through a population of particles
         (or candidates) and find those with the highest fitness as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        n_particle: integer value that determines the initial population size of particles

        vel: a value that determines the velocity of particles

        coeff_p: a value in the range 0-1, personal coefficient

        coeff_g: a value in the range 0-1, global coefficient

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}


    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, n_particle=N_PARTICLES, vel=VELOCITY, coeff_p=PERSONAL_COEFF,
                 coeff_g=GLOBAL_COEFF):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
            position and fitness. The fitness is derived from the computed support of that candidate, the higher the
            support value the higher the fitness. The aim of the algorithm is search through a population of particles
            (or candidates) and find those with the highest fitness as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            n_particle: integer value that determines the initial population size of particles

            vel: a value that determines the velocity of particles

            coeff_p: a value in the range 0-1, personal coefficient

            coeff_g: a value in the range 0-1, global coefficient

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param n_particle: initial particle population, default is 5
        :param vel: velocity, default is 0.9
        :param coeff_p: personal coefficient, default is 0.01
        :param coeff_g: global coefficient, default is 0.9
        """
        super(ParticleGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        self.n_particles = n_particle
        self.velocity = vel
        self.coeff_p = coeff_p
        self.coeff_g = coeff_g

    def discover(self):
        """Description

        Searches through particle positions to find GP candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        # Prepare data set
        self.fit_bitmap()

        # self.target = 1
        # self.target_error = 1e-6
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        it_count = 0
        eval_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)

        # Empty particle template
        empty_particle = structure()
        empty_particle.position = None
        empty_particle.fitness = None

        # Initialize Population
        particle_pop = empty_particle.repeat(self.n_particles)
        for i in range(self.n_particles):
            particle_pop[i].position = random.randrange(var_min, var_max)
            particle_pop[i].fitness = 1

        pbest_pop = particle_pop.copy()
        gbest_particle = pbest_pop[0]

        # Best particle (ever found)
        best_particle = empty_particle.deepcopy()
        best_particle.position = gbest_particle.position
        best_particle.fitness = NumericSS.cost_function(best_particle.position, attr_keys, self)

        velocity_vector = np.ones(self.n_particles)
        best_fitness_arr = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(self.n_particles):
                # UPDATED
                if particle_pop[i].position < var_min or particle_pop[i].position > var_max:
                    particle_pop[i].fitness = 1
                else:
                    particle_pop[i].fitness = NumericSS.cost_function(particle_pop[i].position, attr_keys, self)
                    if particle_pop[i].fitness == 1:
                        invalid_count += 1
                    eval_count += 1
                    str_eval += "{}: {} \n".format(eval_count, particle_pop[i].fitness)

                if pbest_pop[i].fitness > particle_pop[i].fitness:
                    pbest_pop[i].fitness = particle_pop[i].fitness
                    pbest_pop[i].position = particle_pop[i].position

                if gbest_particle.fitness > particle_pop[i].fitness:
                    gbest_particle.fitness = particle_pop[i].fitness
                    gbest_particle.position = particle_pop[i].position
            # if abs(gbest_fitness_value - self.target) < self.target_error:
            #    break
            if best_particle.fitness > gbest_particle.fitness:
                best_particle = gbest_particle.deepcopy()

            for i in range(self.n_particles):
                new_velocity = (self.velocity * velocity_vector[i]) + \
                               (self.coeff_p * random.random()) * (pbest_pop[i].position - particle_pop[i].position) + \
                               (self.coeff_g * random.random()) * (gbest_particle.position - particle_pop[i].position)
                particle_pop[i].position = particle_pop[i].position + new_velocity

            best_gp = NumericSS.decode_gp(attr_keys, best_particle.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_particle.fitness = 1

            try:
                # Show Iteration Information
                best_fitness_arr[it_count] = best_particle.fitness
                str_iter += "{}: {} \n".format(it_count, best_particle.fitness)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "PSO-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns

        return out


class RandomGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Random Search Algorithm (LS-GRAANK)
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
         associated with it. The cost is derived from the computed support of that candidate, the higher the support
         value the lower the cost. The aim of the algorithm is search through group of positions and find those with
         the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.RandomGRAANK(dummy_df, 0.5, max_iter=3)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "RS-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 1,
    "Iterations": 3}

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Random Search Algorithm (LS-GRAANK)
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, we assume that every GP candidate may be represented as a position that has a cost value
            associated with it. The cost is derived from the computed support of that candidate, the higher the support
            value the lower the cost. The aim of the algorithm is search through group of positions and find those with
            the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.RandomGRAANK(dummy_df, 0.5, max_iter=3)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "RS-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 1,
        "Iterations": 3}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        """
        super(RandomGRAANK, self).__init__(*args)
        self.max_iteration = max_iter

    def discover(self):
        """Description

        Uses random search to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """
        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Parameters
        it_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        eval_count = 0

        # Empty Individual Template
        candidate = structure()
        candidate.position = None
        candidate.cost = float('inf')

        # INITIALIZE
        best_sol = candidate.deepcopy()
        best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        repeated = 0
        invalid_count = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            candidate.position = ((var_min + random.random()) * (var_max - var_min))
            NumericSS.apply_bound(candidate, var_min, var_max)
            candidate.cost = NumericSS.cost_function(candidate.position, attr_keys, self)
            if candidate.cost == 1:
                invalid_count += 1

            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_sol.cost = 1

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "RS-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out
