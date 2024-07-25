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
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input

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

    def discover_tgp(self, parallel=False):
        """"""

        # 1. Compute mutual information
        self.compute_mutual_info()

        # 2. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(self.mi_arr, self.initial_mutual_info))
        absolute_error = np.sqrt(squared_diff)
        optimal_steps_arr = np.argmin(absolute_error, axis=0)
        max_step = (np.max(optimal_steps_arr) + 1)
        print(f"Largest step delay: {max_step}\n")
        # print(f"Abs.E.: {absolute_error}\n")

        # 3. Integrate feature indices with the computed steps
        # optimal_dict = dict(map(lambda key, val: (int(key), int(val+1)), self.feature_cols, optimal_steps_arr))
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        print(f"Optimal Dict: {optimal_dict}\n")

        # 4. Create final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)
        # print(f"{delayed_data}\n")
        print(f"{time_data}\n")

        # 5. Build triangular MF
        a, b, c = TGradAMI.build_mf(time_data)
        self.tri_mf_data = np.array([a, b, c])
        print(f"{a}, {b}, {c}")

        # 6. Learn the best MF through slide-descent/sliding
        # 7. Apply cartesian product on multiple MFs to pick the MF with the biggest center (inference logic)

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
        return a, b, c

    @staticmethod
    def learn_best_mf(a: float, b: float, c: float, x_train: np.ndarray):
        """"""
        # if a <= x <= b then y_hat = (x - a) / (b - a)
        # if b <= x <= c then y_hat = (c - x) / (c - b)
        # Initialize parameters
        min_membership = 0.001

        # 1. ML Approach
        tri_mf_data = np.array([a, b, c])
        y_train = np.full_like(x_train, 1)

        model = Sequential([
            Input(shape=(1,)),
            FixedWeightsLayer(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        model.compile(optimizer='adam', loss=TGradAMI.cost_function_wrapper(tri_mf_data, min_membership))
        model.fit(x_train, y_train, epochs=10)

        weights = model.layers[0].get_weights()[0]
        bias = model.layers[0].get_weights()[1]
        print(f"weights: {weights}")
        print(f"bias: {bias}")

        # 2. Manual Approach
        # candidate_steps = [float((((0.5*(b+a)/x)-1) if (x <= b) else float((0.5*(b+c)/x)-1)) * x) for x in x_data]
        # print(f"Candidate Steps: {candidate_steps}")
        # for i in range(10):
        #    print(f"Slide: {i}")
        #    print(f"x-train: {x_train}")
        #    # 1. Generate fuzzy data set using MF from x_data
        #    # Method 1 (OK)
        #    x_hat = np.where(x_train <= b, (x_train-a)/(b-a), (c-x_train)/(c-b))
        #    b = 0.5
        #    x_train = x_train + b
        #    # 2. Generate y_train based on the given criteria (x>0.5)
        #    y_hat = np.where((x_hat >= min_membership), 1, 0)
        #    print(f"x-hat: {x_hat}")
        #    print(f"y-hat: {y_hat}\n")

    @staticmethod
    def cost_function_wrapper(tri_mf: np.ndarray, min_membership: float):
        """
        Computes the logistic regression cost function for a fuzzy set created from a triangular membership function.

        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
        :return: A numpy array of the cost function values.
        """

        def custom_loss(y_true: np.ndarray, x_hat: np.ndarray):
            """
                Computes the logistic regression cost function for a fuzzy set created from a triangular membership function.

                :param y_true: A numpy array of the true labels.
                :param x_hat: A numpy array of the predicted labels.
                :return: A numpy array of the cost function values.
            """

            # 1. Generate fuzzy data set using MF from x_data
            a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]
            x_hat = np.where(x_hat <= b, (x_hat - a) / (b - a), (c - x_hat) / (c - b))

            # 2. Generate y_train based on the given criteria (x>minimum_membership)
            y_hat = np.where((x_hat >= min_membership), 1, 0)

            cost = -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
            return cost
        return custom_loss


class FixedWeightsLayer(Layer):
    def __init__(self, units, **kwargs):
        super(FixedWeightsLayer, self).__init__(**kwargs)
        self.fixed_weights = None
        self.bias = None
        self.units = units

    def build(self, input_shape):
        # Initialize bias as a trainable variable
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        # Set weights to 1 and make them non-trainable
        self.fixed_weights = tf.ones((input_shape[-1], self.units), dtype=tf.float32)

    def call(self, inputs):
        return tf.matmul(inputs, self.fixed_weights) + self.bias
